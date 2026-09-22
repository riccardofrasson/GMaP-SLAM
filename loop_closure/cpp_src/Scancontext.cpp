#include "Scancontext.h"
#include <iostream> // Mantenuto per stampe errori
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <exception> // Per std::out_of_range
#include <limits> // Per std::numeric_limits

// Usiamo il namespace definito nei nostri header
namespace radar_graph_slam {

// Funzioni helper (le lasciamo fuori dalla classe)
float rad2deg(float radians) { return radians * 180.0 / M_PI; }
float deg2rad(float degrees) { return degrees * M_PI / 180.0; }

float xy2theta( const float & _x, const float & _y )
{
    // Usa atan2(y, x) per robustezza e correttezza [-180, 180]
    return std::atan2(_y, _x) * 180.0 / M_PI;
}

MatrixXd circshift( MatrixXd &_mat, int _num_shift )
{
    // Controllo sicurezza: _num_shift negativo non gestito da %
     if (_mat.cols() == 0) return _mat; // Controllo per matrice vuota
    if (_num_shift < 0) {
         _num_shift = (_mat.cols() + (_num_shift % _mat.cols())) % _mat.cols();
    } else {
         _num_shift = _num_shift % _mat.cols();
    }
    if( _num_shift == 0 ) return _mat;


    MatrixXd shifted_mat = MatrixXd::Zero( _mat.rows(), _mat.cols() );
    for ( int col_idx = 0; col_idx < _mat.cols(); col_idx++ )
    {
        int new_location = (col_idx + _num_shift) % _mat.cols();
        shifted_mat.col(new_location) = _mat.col(col_idx);
    }
    return shifted_mat;
}

std::vector<float> eig2stdvec( MatrixXd _eigmat )
{
     // Controllo sicurezza
     if (_eigmat.size() == 0) return {};
    std::vector<float> vec( _eigmat.data(), _eigmat.data() + _eigmat.size() );
    return vec;
}

// Implementazioni dei metodi della classe SCManager

// Costruttore senza stampe debug
SCManager::SCManager() {
     // Calcola subito PC_UNIT_SECTOR_ANGLE con controlli
     setAzimuthRange(PC_AZIMUTH_ANGLE_MAX); // Usa il valore max per impostare min e unit
}


void SCManager::setScDistThresh(double thresh) {
     SC_DIST_THRES = thresh;
 }
void SCManager::setAzimuthRange(double range) {
    PC_AZIMUTH_ANGLE_MAX = range;
    PC_AZIMUTH_ANGLE_MIN = -range;
    // Assicura che il denominatore non sia zero
    if (PC_NUM_SECTOR > 0) {
        double angle_range = PC_AZIMUTH_ANGLE_MAX - PC_AZIMUTH_ANGLE_MIN;
        if (angle_range <= 1e-5) { // Usa tolleranza per float
             std::cerr << "[ERRORE] Range angolare non valido (<=0) in setAzimuthRange!" << std::endl;
             PC_UNIT_SECTOR_ANGLE = 0;
        } else {
             PC_UNIT_SECTOR_ANGLE = angle_range / double(PC_NUM_SECTOR);
        }
    } else {
         PC_UNIT_SECTOR_ANGLE = 0; // O gestisci l'errore
         std::cerr << "[ERRORE] PC_NUM_SECTOR è zero in setAzimuthRange!" << std::endl;
    }
     if (PC_NUM_RING <= 0) {
          std::cerr << "[ERRORE] PC_NUM_RING non positivo in setAzimuthRange!" << std::endl;
     }
}

double SCManager::distDirectSC ( MatrixXd &_sc1, MatrixXd &_sc2 )
{
    int num_eff_cols = 0;
    double sum_sector_similarity = 0;
    for ( int col_idx = 0; col_idx < _sc1.cols(); col_idx++ )
    {
        const auto& col_sc1 = _sc1.col(col_idx);
        const auto& col_sc2 = _sc2.col(col_idx);
        double norm1 = col_sc1.norm();
        double norm2 = col_sc2.norm();
        if( norm1 < 1e-6 || norm2 < 1e-6 ) continue;
        double dot_product = col_sc1.dot(col_sc2);
        double norm_product = norm1 * norm2;
         if (norm_product < 1e-9) continue;
        double sector_similarity = dot_product / norm_product;
        sector_similarity = std::max(-1.0, std::min(1.0, sector_similarity));
        sum_sector_similarity += sector_similarity;
        num_eff_cols++;
    }
    double sc_sim = (num_eff_cols == 0) ? 0.0 : sum_sector_similarity / num_eff_cols;
    return std::max(0.0, 1.0 - sc_sim); // Assicura >= 0
}

int SCManager::fastAlignUsingVkey( MatrixXd & _vkey1, MatrixXd & _vkey2)
{
     if (_vkey1.cols() == 0 || _vkey1.cols() != _vkey2.cols()) {
          std::cerr << "[ERRORE] Dimensioni VKey non valide in fastAlignUsingVkey!" << std::endl;
          return 0;
     }
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = std::numeric_limits<double>::max();
    for ( int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++ )
    {
        MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);
        MatrixXd vkey_diff = _vkey1 - vkey2_shifted;
        double cur_diff_norm = vkey_diff.norm();
        if( cur_diff_norm < min_veky_diff_norm )
        {
            argmin_vkey_shift = shift_idx;
            min_veky_diff_norm = cur_diff_norm;
        }
    }
    return argmin_vkey_shift;
}

std::pair<double, int> SCManager::distanceBtnScanContext( MatrixXd &_sc1, MatrixXd &_sc2 )
{
     if (_sc1.rows() != PC_NUM_RING || _sc1.cols() != PC_NUM_SECTOR ||
         _sc2.rows() != PC_NUM_RING || _sc2.cols() != PC_NUM_SECTOR) {
          std::cerr << "[ERRORE] Dimensioni Scan Context (" << _sc1.rows() << "x" << _sc1.cols() << ", "
                    << _sc2.rows() << "x" << _sc2.cols() << ") non corrispondenti ai parametri ("
                    << PC_NUM_RING << "x" << PC_NUM_SECTOR << ")!" << std::endl;
          return make_pair(1.0, 0);
     }
     if (_sc1.cols() <= 0) {
          std::cerr << "[ERRORE] Scan Context con 0 colonne!" << std::endl;
          return make_pair(1.0, 0);
     }

    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext( _sc1 );
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext( _sc2 );
    int argmin_vkey_shift = fastAlignUsingVkey( vkey_sc1, vkey_sc2 );

    const int SEARCH_RADIUS = round( 0.5 * SEARCH_RATIO * _sc1.cols() );
    std::vector<int> shift_idx_search_space { argmin_vkey_shift };
    for ( int ii = 1; ii < SEARCH_RADIUS + 1; ii++ )
    {
        shift_idx_search_space.push_back( (argmin_vkey_shift + ii + _sc1.cols()) % _sc1.cols() );
        shift_idx_search_space.push_back( (argmin_vkey_shift - ii + _sc1.cols()) % _sc1.cols() );
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());
    shift_idx_search_space.erase( std::unique( shift_idx_search_space.begin(), shift_idx_search_space.end() ), shift_idx_search_space.end() );

    int argmin_shift = 0;
    double min_sc_dist = std::numeric_limits<double>::max();
    for ( int num_shift: shift_idx_search_space )
    {
         if (num_shift < 0 || num_shift >= _sc1.cols()){
              std::cerr << "[ERRORE] num_shift invalido (" << num_shift << ") in distanceBtnScanContext!" << std::endl;
              continue;
         }
        MatrixXd sc2_shifted = circshift(_sc2, num_shift);
        double cur_sc_dist = distDirectSC( _sc1, sc2_shifted );
        if( cur_sc_dist < min_sc_dist )
        {
            argmin_shift = num_shift;
            min_sc_dist = cur_sc_dist;
        }
    }
    min_sc_dist = std::max(0.0, min_sc_dist);
    return make_pair(min_sc_dist, argmin_shift);
}

// ----- FUNZIONE makeScancontext PULITA -----
MatrixXd SCManager::makeScancontext( pcl::PointCloud<SCPointType> & _scan_down )
{
    const int NO_POINT = -1000;
    // Controllo sicurezza parametri critici
    if (PC_NUM_RING <= 0 || PC_NUM_SECTOR <= 0 || PC_MAX_RADIUS <= 1e-3 || PC_UNIT_SECTOR_ANGLE <= 1e-6) {
         std::cerr << "[ERRORE CRITICO] Parametri SCManager non validi rilevati in makeScancontext!" << std::endl;
         return MatrixXd::Zero(0,0); // Ritorna matrice vuota
    }

    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    for (const auto& pt_orig : _scan_down.points)
    {
        SCPointType pt;
        pt.x = pt_orig.x;
        pt.y = pt_orig.y;
        pt.z = pt_orig.z + LIDAR_HEIGHT;
        pt.intensity = pt_orig.intensity;

        // Calcola range e angolo
        float azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
        float azim_angle = (std::atan2(pt.x, pt.y) - M_PI_2) * 180.0 / M_PI;

        // Filtra per range e angolo
        if (azim_range < 0.1 || azim_range > PC_MAX_RADIUS) continue;
        if (azim_angle < PC_AZIMUTH_ANGLE_MIN || azim_angle > PC_AZIMUTH_ANGLE_MAX) continue;

        // Calcola indici (base 1)
        int ring_idx = static_cast<int>(std::ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING));
        ring_idx = std::max(1, std::min(PC_NUM_RING, ring_idx));

        float relative_angle = azim_angle - PC_AZIMUTH_ANGLE_MIN;
        int sctor_idx = static_cast<int>(std::ceil(relative_angle / PC_UNIT_SECTOR_ANGLE));
        sctor_idx = std::max(1, std::min(PC_NUM_SECTOR, sctor_idx));

        // Controllo robusto limiti prima dell'accesso (base 0 per accesso)
        int row = ring_idx - 1;
        int col = sctor_idx - 1;

        if (row >= 0 && row < desc.rows() && col >= 0 && col < desc.cols()) {
             if (desc(row, col) < pt.intensity || desc(row, col) == NO_POINT)
                desc(row, col) = pt.intensity;
        } else {
             std::cerr << "[ERRORE] Accesso Matrice Imprevisto! desc(" << row << ", " << col << ")" << std::endl;
        }
    }

    // Resetta i bin vuoti a 0
    for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
        for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
            if(desc(row_idx, col_idx) == NO_POINT)
                desc(row_idx, col_idx) = 0;

    return desc;
}

MatrixXd SCManager::makeRingkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    if (_desc.rows() == 0) return MatrixXd::Zero(0,1);
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);
    for ( int row_idx = 0; row_idx < _desc.rows(); row_idx++ )
    {
        Eigen::MatrixXd curr_row = _desc.row(row_idx);
        invariant_key(row_idx, 0) = curr_row.mean();
    }
    return invariant_key;
}

MatrixXd SCManager::makeSectorkeyFromScancontext( Eigen::MatrixXd &_desc )
{
    if (_desc.cols() == 0) return MatrixXd::Zero(1,0);
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for ( int col_idx = 0; col_idx < _desc.cols(); col_idx++ )
    {
        Eigen::MatrixXd curr_col = _desc.col(col_idx);
        variant_key(0, col_idx) = curr_col.mean();
    }
    return variant_key;
}

// ----- FUNZIONE makeAndSaveScancontextAndKeys PULITA -----
void SCManager::makeAndSaveScancontextAndKeys( pcl::PointCloud<SCPointType> & _scan_down )
{
    Eigen::MatrixXd sc = makeScancontext(_scan_down);
    // Controllo robusto se makeScancontext ha ritornato una matrice valida
    if (sc.rows() != PC_NUM_RING || sc.cols() != PC_NUM_SECTOR) {
         std::cerr << "[ERRORE CRITICO] makeScancontext ha ritornato matrice errata ("
                   << sc.rows() << "x" << sc.cols() << ")!" << std::endl;
         throw std::runtime_error("makeScancontext failed");
    }
    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
}

const Eigen::MatrixXd& SCManager::getConstRefRecentSCD(void)
{
    if (polarcontexts_.empty()) {
        std::cerr << "[ERRORE] getConstRefRecentSCD chiamato su database vuoto!" << std::endl;
        throw std::out_of_range("Database Scan Context vuoto");
    }
    return polarcontexts_.back();
}


// ----- FUNZIONE detectLoopClosureID PULITA -----
std::pair<int, float> SCManager::detectLoopClosureID (const std::vector<KeyFramePtr>& candidate_keyframes, const KeyFramePtr& new_keyframe)
{
    int loop_id = -1; // Indice globale del loop trovato

    // Controllo sicurezza indice keyframe corrente
    if (new_keyframe == nullptr || new_keyframe->index < 0 || (size_t)new_keyframe->index >= polarcontext_invkeys_mat_.size() || (size_t)new_keyframe->index >= polarcontexts_.size()) {
        std::cerr << "[ERRORE] Indice new_keyframe (" << (new_keyframe ? std::to_string(new_keyframe->index) : "NULL") << ") invalido!" << std::endl;
        return make_pair(loop_id, 0.0f);
    }

    // Controllo età keyframe corrente
    if (new_keyframe->index < NUM_EXCLUDE_RECENT) {
        return make_pair(loop_id, 0.0f);
    }

    const auto& curr_key = polarcontext_invkeys_mat_.at(new_keyframe->index);
    const auto& curr_desc = polarcontexts_.at(new_keyframe->index);

    // 1. Filtra candidati validi (controllo indici e età)
    std::vector<size_t> searchable_global_indices;
    for (const auto& ckf : candidate_keyframes) {
         if (ckf == nullptr || ckf->index < 0 || (size_t)ckf->index >= polarcontext_invkeys_mat_.size() || (size_t)ckf->index >= polarcontexts_.size()) {
              std::cerr << "[WARNING] Indice candidato (" << (ckf ? std::to_string(ckf->index) : "NULL") << ") invalido o fuori limiti! Skip." << std::endl;
              continue;
          }
        if (new_keyframe->index - ckf->index >= NUM_EXCLUDE_RECENT) {
            searchable_global_indices.push_back(ckf->index);
        }
    }

    if (searchable_global_indices.empty()) {
        return make_pair(loop_id, 0.0f);
    }

    // 2. Ricostruisci l'albero di ricerca (se necessario)
    bool rebuild_tree = (tree_making_period_conter % TREE_MAKING_PERIOD_ == 0);
    if (rebuild_tree)
    {
        polarcontext_invkeys_to_search_.clear();
        polarcontext_invkeys_to_search_.reserve(searchable_global_indices.size());
        for (size_t global_idx : searchable_global_indices) {
             if (global_idx < polarcontext_invkeys_mat_.size()) { // Controllo sicurezza ridondante
                 polarcontext_invkeys_to_search_.push_back(polarcontext_invkeys_mat_.at(global_idx));
             }
        }

        if (!polarcontext_invkeys_to_search_.empty()) {
            try {
                 polarcontext_tree_ = std::make_unique<InvKeyTree>(PC_NUM_RING, polarcontext_invkeys_to_search_, 10);
            } catch (const std::exception& e) {
                 std::cerr << "[ERRORE] Eccezione creazione albero: " << e.what() << std::endl;
                 polarcontext_tree_.reset();
                 return make_pair(loop_id, 0.0f);
            }
        } else {
             polarcontext_tree_.reset();
        }
    }
    tree_making_period_conter++;

    if (!polarcontext_tree_) {
        return make_pair(loop_id, 0.0f);
    }

    // 3. Cerca nell'albero
    double min_dist = std::numeric_limits<double>::max();
    int nn_align = 0;
    int nn_idx = -1; // Indice globale del miglior match

    size_t num_elements_in_tree = polarcontext_invkeys_to_search_.size();
    size_t num_to_search = std::min((size_t)NUM_CANDIDATES_FROM_TREE, num_elements_in_tree);

    if (num_to_search == 0) {
         return make_pair(loop_id, 0.0f);
    }

    std::vector<size_t> knn_indices(num_to_search);
    std::vector<float> out_dists_sqr(num_to_search);
    nanoflann::KNNResultSet<float> knnsearch_result(num_to_search);
    knnsearch_result.init(&knn_indices[0], &out_dists_sqr[0]);

    try {
        polarcontext_tree_->index->findNeighbors(knnsearch_result, &curr_key[0], nanoflann::SearchParameters(10));
    } catch (const std::exception& e) {
        std::cerr << "[ERRORE] Eccezione findNeighbors: " << e.what() << std::endl;
        return make_pair(loop_id, 0.0f);
    }
    size_t num_results = knnsearch_result.size();

    // 4. Calcola distanza precisa
    for (size_t i = 0; i < num_results; ++i)
    {
        size_t local_idx = knn_indices[i];
        if (local_idx >= searchable_global_indices.size()) {
            std::cerr << "[ERRORE] Indice locale k-NN (" << local_idx << ") fuori limiti! Skip." << std::endl;
            continue;
        }
        size_t global_candidate_idx = searchable_global_indices[local_idx];
        if (global_candidate_idx >= polarcontexts_.size()) {
             std::cerr << "[ERRORE] Indice globale (" << global_candidate_idx << ") fuori limiti polarcontexts_! Skip." << std::endl;
             continue;
        }

        const auto& polarcontext_candidate = polarcontexts_.at(global_candidate_idx);
        std::pair<double, int> sc_dist_result = distanceBtnScanContext(const_cast<MatrixXd&>(curr_desc), const_cast<MatrixXd&>(polarcontext_candidate));

        double candidate_dist = sc_dist_result.first;
        int candidate_align = sc_dist_result.second;

        if (candidate_dist < min_dist)
        {
            min_dist = candidate_dist;
            nn_align = candidate_align;
            nn_idx = global_candidate_idx;
        }
    }

    // Dichiara yaw_diff_rad qui
    float yaw_diff_rad = 0.0f;

    // 5. Verifica soglia
    if (nn_idx != -1 && min_dist < SC_DIST_THRES)
    {
        loop_id = nn_idx;
        // Stampa solo se trova loop
        cout << "\033[32m [Loop found] Nearest SC distance: " << min_dist << " between " << new_keyframe->index << " and " << loop_id << ". \033[0m" << endl;

         if (PC_UNIT_SECTOR_ANGLE <= 1e-6) {
             std::cerr << "[ERRORE] PC_UNIT_SECTOR_ANGLE non valido (" << PC_UNIT_SECTOR_ANGLE << "), impossibile calcolare yaw!" << std::endl;
         } else {
             yaw_diff_rad = deg2rad(nn_align * PC_UNIT_SECTOR_ANGLE);
         }
    }

    return make_pair(loop_id, yaw_diff_rad);
}

} // namespace radar_graph_slam