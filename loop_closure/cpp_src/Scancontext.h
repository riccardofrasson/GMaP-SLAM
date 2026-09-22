#pragma once

#include <vector>
#include <iostream>
#include <memory>
#include <utility>
#include <Eigen/Dense>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "nanoflann.hpp"
#include "KDTreeVectorOfVectorsAdaptor.h"
// #include "tictoc.h" // Commentato per rimuovere dipendenza non essenziale
#include "types.h" // La nostra definizione di KeyFrame

namespace radar_graph_slam {

using namespace Eigen;
using namespace nanoflann;

using std::cout;
using std::endl;
using std::make_pair;

using SCPointType = pcl::PointXYZI;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// Funzioni helper
float xy2theta(const float& _x, const float& _y);
MatrixXd circshift(MatrixXd& _mat, int _num_shift);
std::vector<float> eig2stdvec(MatrixXd _eigmat);


class SCManager
{
public:
    // SCManager() = default; // RIMOSSO "= default;"
    SCManager(); // Dichiarazione semplice del costruttore

    void setScDistThresh(double thresh);
    void setAzimuthRange(double range);

    Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType>& _scan_down);
    Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd& _desc);
    Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd& _desc);

    int fastAlignUsingVkey(MatrixXd& _vkey1, MatrixXd& _vkey2);
    double distDirectSC(MatrixXd& _sc1, MatrixXd& _sc2);
    std::pair<double, int> distanceBtnScanContext(MatrixXd& _sc1, MatrixXd& _sc2);

    void makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType>& _scan_down);
    std::pair<int, float> detectLoopClosureID(const std::vector<KeyFramePtr>& candidate_keyframes, const KeyFramePtr& new_keyframe);
    const Eigen::MatrixXd& getConstRefRecentSCD(void);

public:
    // Parametri (valori di default originali)
    double LIDAR_HEIGHT = 1.2;
    double PC_AZIMUTH_ANGLE_MAX = 56.5;
    double PC_AZIMUTH_ANGLE_MIN = -56.5;
    int    PC_NUM_RING = 40;
    int    PC_NUM_SECTOR = 20; // Originale: 20
    double PC_MAX_RADIUS = 80.0;
    // Ricalcola angolo unitario basato su 20 settori
    double PC_UNIT_SECTOR_ANGLE = (56.5 - (-56.5)) / 20.0;
    int    NUM_EXCLUDE_RECENT = 10;
    int    NUM_CANDIDATES_FROM_TREE = 3; // Originale: 3
    double SEARCH_RATIO = 0.1; // Originale: 0.1
    double SC_DIST_THRES = 0.5; // Originale: 0.5
    int    TREE_MAKING_PERIOD_ = 10; // Originale: Ricostruzione periodica
    int tree_making_period_conter = 0; // Contatore per ricostruzione periodica

    // Dati interni
    std::vector<Eigen::MatrixXd> polarcontexts_;
    std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
    std::vector<Eigen::MatrixXd> polarcontext_vkeys_;
    KeyMat polarcontext_invkeys_mat_;
    KeyMat polarcontext_invkeys_to_search_;
    std::unique_ptr<InvKeyTree> polarcontext_tree_;
};

} // namespace radar_graph_slam