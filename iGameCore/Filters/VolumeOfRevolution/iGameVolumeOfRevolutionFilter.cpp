#include "iGameVolumeOfRevolutionFilter.h"
#include "iGameArrayObject.h"
#include "iGameCellArray.h"
#include "iGamePoints.h"
#include "iGameStructuredMesh.h"
#include "iGameSurfaceMesh.h"
#include "iGameUnstructuredMesh.h"
#include "iGameVector.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <set>
#include <vector>
IGAME_NAMESPACE_BEGIN

// ---------- 辅助函数 ----------
static void BuildAdjacency(const std::vector<Edge>& edges, std::map<IGsize, std::set<IGsize>>& adj) {
    for (const auto& e: edges) {
        adj[e.v0].insert(e.v1);
        adj[e.v1].insert(e.v0);
    }
}

// 三角扇生成圆盘端盖
static void AddCircularCap(IGsize centerPtIdx, const std::vector<IGsize>& ringPts, CellArray* cells,
                           UnsignedIntArray* types) {
    if (ringPts.size() < 3) return;
    for (size_t i = 0; i < ringPts.size(); ++i) {
        size_t j = (i + 1) % ringPts.size();
        igIndex tri[3] = {static_cast<igIndex>(centerPtIdx), static_cast<igIndex>(ringPts[i]),
                          static_cast<igIndex>(ringPts[j])};
        cells->AddCellIds(tri, 3);
        types->AddValue(IG_TRIANGLE);
    }
}

// ---------- 主执行函数 ----------
bool VolumeOfRevolutionFilter::Execute() {
    DataObject::Pointer input = GetInput(0);
    if (!input) return false;

    std::vector<Vector3d> contourPts; // 所有轮廓点的坐标
    std::vector<Edge> edges;          // 轮廓线段（无序）

    IGenum type = input->GetDataObjectType();

    if (type == IG_UNSTRUCTURED_MESH) {
        auto inMesh = DynamicCast<UnstructuredMesh>(input);
        if (!inMesh) {
            igError("Failed to cast to UnstructuredMesh.");
            return false;
        }
        auto inPoints = inMesh->GetPoints();
        if (!inPoints || inPoints->GetNumberOfPoints() < 2) {
            igError("UnstructuredMesh has insufficient points.");
            return false;
        }
        // 提取点坐标
        IGsize numPts = inPoints->GetNumberOfPoints();
        contourPts.resize(numPts);
        for (IGsize i = 0; i < numPts; ++i) { inPoints->GetPoint(i, contourPts[i]); }
        // 提取线段单元
        auto cells = inMesh->GetCells();
        auto types = inMesh->GetCellTypes();
        if (cells && types) {
            IGsize numCells = cells->GetNumberOfCells();
            for (IGsize i = 0; i < numCells; ++i) {
                if (types->GetValue(i) == IG_LINE) {
                    igIndex ids[2];
                    if (cells->GetCellIds(i, ids) == 2) {
                        edges.push_back({static_cast<IGsize>(ids[0]), static_cast<IGsize>(ids[1])});
                    }
                }
            }
        }
        if (edges.empty() && numPts > 1) {
            for (IGsize i = 0; i < numPts - 1; ++i) { edges.push_back({i, i + 1}); }
        }
    } else if (type == IG_SURFACE_MESH) {
        auto surf = DynamicCast<SurfaceMesh>(input);
        surf->RequestEditStatus();
        if (!surf) {
            igError("Failed to cast to SurfaceMesh.");
            return false;
        }
        // 提取点坐标
        auto pts = surf->GetPoints();
        if (!pts || pts->GetNumberOfPoints() < 3) {
            igError("SurfaceMesh has insufficient points.");
            return false;
        }
        IGsize numPts = pts->GetNumberOfPoints();
        contourPts.resize(numPts);
        for (IGsize i = 0; i < numPts; ++i) { pts->GetPoint(i, contourPts[i]); }

        // 提取边界边：遍历所有边，保留 IsBoundaryEdge 为 true 的边
        IGsize numEdges = surf->GetNumberOfEdges();
        for (IGsize eid = 0; eid < numEdges; ++eid) {
            if (surf->IsBoundaryEdge(eid)) {
                igIndex ids[2];
                if (surf->GetEdgePointIds(eid, ids) == 2) {
                    edges.push_back({static_cast<IGsize>(ids[0]), static_cast<IGsize>(ids[1])});
                }
            }
        }
        if (edges.empty()) {
            igError("SurfaceMesh has no boundary edges (not a valid 2D contour).");
            return false;
        }
    } else if (type == IG_STRUCTURED_MESH) {
        auto sMesh = DynamicCast<StructuredMesh>(input);
        if (!sMesh) {
            igError("Failed to cast to StructuredMesh.");
            return false;
        }
        auto pts = sMesh->GetPoints();
        if (!pts || pts->GetNumberOfPoints() < 2) {
            igError("StructuredMesh has insufficient points.");
            return false;
        }

        // 获取维度大小（点数）
        igIndex* dim = sMesh->GetDimensionSize(); // 返回{ni, nj, nk}
        IGsize ni = dim[0], nj = dim[1], nk = dim[2];

        // 提取所有点坐标
        IGsize expectedPts = ni * nj * nk;
        IGsize actualPts = pts->GetNumberOfPoints();
        if (actualPts != expectedPts) {
            igError("StructuredMesh point count does not match dimensions.");
            return false;
        }
        contourPts.resize(actualPts);
        for (IGsize idx = 0; idx < actualPts; ++idx) { pts->GetPoint(idx, contourPts[idx]); }

        // 辅助：根据 (i,j,k) 计算线性索引
        auto idxOf = [&](IGsize i, IGsize j, IGsize k) -> IGsize { return i + j * ni + k * ni * nj; };

        // 用 set 去重（避免四条边界相交处重复）
        std::set<std::pair<IGsize, IGsize>> edgeSet;

        auto addEdge = [&](IGsize a, IGsize b) {
            if (a > b) std::swap(a, b);
            edgeSet.insert({a, b});
        };

        if (nj == 1 && nk == 1) {
            // 1D 曲线：沿 i 方向
            for (IGsize i = 0; i < ni - 1; ++i) { addEdge(idxOf(i, 0, 0), idxOf(i + 1, 0, 0)); }
        } else if (nk == 1) {
            // 2D 面片 (i, j 平面)：提取四条边界
            // 底边 j=0
            for (IGsize i = 0; i < ni - 1; ++i) addEdge(idxOf(i, 0, 0), idxOf(i + 1, 0, 0));
            // 顶边 j=nj-1
            for (IGsize i = 0; i < ni - 1; ++i) addEdge(idxOf(i, nj - 1, 0), idxOf(i + 1, nj - 1, 0));
            // 左边 i=0
            for (IGsize j = 0; j < nj - 1; ++j) addEdge(idxOf(0, j, 0), idxOf(0, j + 1, 0));
            // 右边 i=ni-1
            for (IGsize j = 0; j < nj - 1; ++j) addEdge(idxOf(ni - 1, j, 0), idxOf(ni - 1, j + 1, 0));
        } else {
            igError("StructuredMesh must be 1D curve or 2D surface (nk==1).");
            return false;
        }

        // 将 set 转为 vector
        for (const auto& e: edgeSet) { edges.push_back({e.first, e.second}); }
        if (edges.empty()) {
            igError("StructuredMesh boundary extraction failed.");
            return false;
        }
    } else {
        igError("iGameVolumeOfRevolution does not support this data type.");
        return false;
    }

    // 检查轮廓是否有效
    if (contourPts.size() < 2 || edges.empty()) {
        igError("No valid contour points or edges found.");
        return false;
    }

    // -------- 2. 旋转轴归一化 --------
    Vector3d axisDir = m_AxisDirection;
    double len = axisDir.norm();
    if (len < 1e-12) {
        igError("Axis direction is zero.");
        return false;
    }
    axisDir /= len;
    Vector3d axisPt = m_AxisPoint;

    // -------- 3. 预计算所有轮廓点的投影 --------
    IGsize numPts = contourPts.size();
    std::vector<PointProjection> proj(numPts);
    for (IGsize i = 0; i < numPts; ++i) {
        Vector3d v = contourPts[i] - axisPt;
        double h = v.dot(axisDir);
        Vector3d v_par = h * axisDir;
        Vector3d v_perp = v - v_par;
        proj[i].r = v_perp.norm();
        proj[i].h = h;
        proj[i].v_perp = v_perp;
    }

    // -------- 4. 角度参数 --------
    double angleStep = m_Angle / m_Resolution;
    const double PI = 3.141592653589793;
    int numTheta = m_Resolution + 1;

    // -------- 5. 生成旋转点云 --------
    auto newPoints = Points::New();
    std::vector<std::vector<IGsize>> pointIndices(numPts);
    for (auto& vec: pointIndices) vec.resize(numTheta);

    for (IGsize iPt = 0; iPt < numPts; ++iPt) {
        double r = proj[iPt].r;
        double h = proj[iPt].h;
        Vector3d v_par = h * axisDir;
        Vector3d v_perp = proj[iPt].v_perp;
        for (int j = 0; j < numTheta; ++j) {
            double theta = j * angleStep;
            Vector3d v_rot;
            if (r < 1e-12) {
                v_rot = Vector3d(0, 0, 0);
            } else {
                Vector3d cross = axisDir.cross(v_perp);
                v_rot = v_perp * std::cos(theta) + cross * std::sin(theta);
            }
            Vector3d newP = axisPt + v_par + v_rot;
            IGsize idx = newPoints->AddPoint(newP[0], newP[1], newP[2]);
            pointIndices[iPt][j] = idx;
        }
    }

    // -------- 6. 生成侧面三角形 --------
    auto newCells = CellArray::New();
    auto newTypes = UnsignedIntArray::New();

    for (const auto& edge: edges) {
        IGsize i0 = edge.v0, i1 = edge.v1;
        for (int j = 0; j < m_Resolution; ++j) {
            int j_next = j + 1;
            IGsize ids[4] = {pointIndices[i0][j], pointIndices[i1][j], pointIndices[i1][j_next],
                             pointIndices[i0][j_next]};
            bool deg0 = (proj[i0].r < 1e-9);
            bool deg1 = (proj[i1].r < 1e-9);
            if (deg0 && deg1) continue;
            else if (deg0) {
                igIndex tri[3] = {static_cast<igIndex>(ids[1]), static_cast<igIndex>(ids[2]),
                                  static_cast<igIndex>(ids[3])};
                newCells->AddCellIds(tri, 3);
                newTypes->AddValue(IG_TRIANGLE);
            } else if (deg1) {
                igIndex tri[3] = {static_cast<igIndex>(ids[0]), static_cast<igIndex>(ids[1]),
                                  static_cast<igIndex>(ids[2])};
                newCells->AddCellIds(tri, 3);
                newTypes->AddValue(IG_TRIANGLE);
            } else {
                igIndex tri1[3] = {static_cast<igIndex>(ids[0]), static_cast<igIndex>(ids[1]),
                                   static_cast<igIndex>(ids[2])};
                newCells->AddCellIds(tri1, 3);
                newTypes->AddValue(IG_TRIANGLE);
                igIndex tri2[3] = {static_cast<igIndex>(ids[0]), static_cast<igIndex>(ids[2]),
                                   static_cast<igIndex>(ids[3])};
                newCells->AddCellIds(tri2, 3);
                newTypes->AddValue(IG_TRIANGLE);
            }
        }
    }

    // -------- 7. 生成端盖 --------
    std::map<double, std::vector<IGsize>> capGroups;
    const double EPS = 1e-9;
    for (const auto& edge: edges) {
        IGsize i0 = edge.v0, i1 = edge.v1;
        Vector3d p0 = contourPts[i0];
        Vector3d p1 = contourPts[i1];
        Vector3d edgeDir = p1 - p0;
        double dot = edgeDir.dot(axisDir);
        if (std::fabs(dot) < EPS) {
            double h = (proj[i0].h + proj[i1].h) * 0.5;
            capGroups[h].push_back(i0);
            capGroups[h].push_back(i1);
        }
    }

    if (!capGroups.empty()) {
        for (auto& group: capGroups) {
            double h = group.first;
            auto& ptIndices = group.second;
            std::set<IGsize> uniquePts(ptIndices.begin(), ptIndices.end());
            if (uniquePts.size() < 2) continue;

            IGsize minIdx = *uniquePts.begin(), maxIdx = *uniquePts.begin();
            double minR = proj[minIdx].r, maxR = proj[maxIdx].r;
            for (IGsize idx: uniquePts) {
                double r = proj[idx].r;
                if (r < minR) {
                    minR = r;
                    minIdx = idx;
                }
                if (r > maxR) {
                    maxR = r;
                    maxIdx = idx;
                }
            }
            if (maxR < 1e-9) continue;

            if (minR < 1e-9) {
                Vector3d center = axisPt + h * axisDir;
                IGsize centerIdx = newPoints->AddPoint(center[0], center[1], center[2]);
                std::vector<IGsize> ringPts;
                ringPts.reserve(numTheta);
                for (int j = 0; j < numTheta; ++j) ringPts.push_back(pointIndices[maxIdx][j]);
                AddCircularCap(centerIdx, ringPts, newCells, newTypes);
            } else {
                std::vector<IGsize> outerRing, innerRing;
                outerRing.reserve(numTheta);
                innerRing.reserve(numTheta);
                for (int j = 0; j < numTheta; ++j) {
                    outerRing.push_back(pointIndices[maxIdx][j]);
                    innerRing.push_back(pointIndices[minIdx][j]);
                }
                for (int j = 0; j < m_Resolution; ++j) {
                    int j_next = (j + 1) % numTheta;
                    igIndex tri1[3] = {static_cast<igIndex>(outerRing[j]), static_cast<igIndex>(innerRing[j]),
                                       static_cast<igIndex>(outerRing[j_next])};
                    newCells->AddCellIds(tri1, 3);
                    newTypes->AddValue(IG_TRIANGLE);
                    igIndex tri2[3] = {static_cast<igIndex>(innerRing[j]), static_cast<igIndex>(innerRing[j_next]),
                                       static_cast<igIndex>(outerRing[j_next])};
                    newCells->AddCellIds(tri2, 3);
                    newTypes->AddValue(IG_TRIANGLE);
                }
            }
        }
    }

    // -------- 8. 输出 --------
    auto outputMesh = UnstructuredMesh::New();
    outputMesh->SetPoints(newPoints);
    outputMesh->SetCells(newCells, newTypes);
    SetOutput(0, outputMesh);
    return true;
}

IGAME_NAMESPACE_END