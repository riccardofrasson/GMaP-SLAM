#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h> // Necessario per array_t

#include "Scancontext.h"
#include <iostream>      // Mantenuto per stampe errori
#include <stdexcept>     // Per std::runtime_error

namespace py = pybind11;
using namespace radar_graph_slam;

// ----- VERSIONE PULITA DI numpyToPcl CHE ACCETTA pybind11::array_t -----
pcl::PointCloud<SCPointType>::Ptr numpyToPcl(const py::array_t<float, py::array::c_style | py::array::forcecast>& points_numpy) {
    py::buffer_info buf_info = points_numpy.request();

    // Controlli di sicurezza sulle dimensioni e tipo
    if (buf_info.ndim != 2) {
        throw std::runtime_error("[ERRORE bindings] L'array NumPy non ha 2 dimensioni!");
    }
    if (buf_info.shape[1] < 3) {
        throw std::runtime_error("[ERRORE bindings] L'array NumPy ha meno di 3 colonne!");
    }
    if (buf_info.format != py::format_descriptor<float>::format()) {
         throw std::runtime_error("[ERRORE bindings] L'array NumPy non è di tipo float32!");
    }

    size_t num_points = buf_info.shape[0];
    size_t num_cols = buf_info.shape[1];

    pcl::PointCloud<SCPointType>::Ptr cloud(new pcl::PointCloud<SCPointType>);
    if (num_points == 0) {
        return cloud; // Ritorna nuvola vuota se input vuoto
    }
    cloud->points.resize(num_points);

    const float* ptr = static_cast<const float*>(buf_info.ptr);

    for (size_t i = 0; i < num_points; ++i) {
        cloud->points[i].x = ptr[i * num_cols + 0]; // Colonna 0
        cloud->points[i].y = ptr[i * num_cols + 1]; // Colonna 1
        cloud->points[i].z = ptr[i * num_cols + 2]; // Colonna 2
        // Usa la quarta colonna come intensità, se esiste
        if (num_cols >= 4) {
            cloud->points[i].intensity = ptr[i * num_cols + 3]; // Colonna 3
        } else {
            cloud->points[i].intensity = 0.0f;
        }
    }
    return cloud;
}

// Funzione principale che definisce il modulo Python
PYBIND11_MODULE(loop_detector, m) {
    m.doc() = "Binding Python per l'algoritmo Scan Context";

    py::class_<SCManager>(m, "SCManager")
        .def(py::init<>())
        .def("set_sc_dist_thresh", &SCManager::setScDistThresh, "Imposta la soglia di distanza")
        .def("set_azimuth_range", &SCManager::setAzimuthRange, "Imposta il range di azimut")

        .def("add_scan", [](SCManager &self, const py::array_t<float, py::array::c_style | py::array::forcecast>& points_numpy) {
            // Verifica preliminare
            if (points_numpy.ndim() != 2 || points_numpy.shape(1) < 3) {
                 std::cerr << "[ERRORE bindings] Dimensioni array NumPy non valide in add_scan lambda!" << std::endl;
                 throw std::runtime_error("Dimensioni array NumPy non valide");
            }

            pcl::PointCloud<SCPointType>::Ptr cloud;
            try {
                 cloud = numpyToPcl(points_numpy);
            } catch (const std::exception& e) {
                 std::cerr << "[ERRORE bindings] Eccezione durante numpyToPcl: " << e.what() << std::endl;
                 throw;
            }

            try {
                if (!cloud) {
                     std::cerr << "[ERRORE bindings] Nuvola PCL nulla dopo conversione!" << std::endl;
                     throw std::runtime_error("Conversione a PCL fallita");
                }
                self.makeAndSaveScancontextAndKeys(*cloud);
            } catch (const std::exception& e) {
                 std::cerr << "[ERRORE bindings] Eccezione C++ durante makeAndSaveScancontextAndKeys: " << e.what() << std::endl;
                 throw; // Rilancia a Python
            } catch (...) {
                 std::cerr << "[ERRORE bindings] Eccezione C++ sconosciuta durante makeAndSaveScancontextAndKeys!" << std::endl;
                 throw;
            }
        }, "Aggiunge nuvola di punti da array NumPy (N, >=3) float32")

        .def("detect_loop", [](SCManager &self, int current_keyframe_index, const std::vector<int>& candidate_indices) -> std::pair<int, float> {
            std::vector<KeyFramePtr> candidate_kfs;
            candidate_kfs.reserve(candidate_indices.size());
            for (int idx : candidate_indices) {
                if (idx < 0) {
                     std::cerr << "[WARNING bindings] Indice candidato negativo (" << idx << ") in detect_loop! Skip." << std::endl;
                     continue;
                }
                auto kf = std::make_shared<KeyFrame>();
                kf->index = idx;
                candidate_kfs.push_back(kf);
            }
            if (current_keyframe_index < 0) {
                 std::cerr << "[ERRORE bindings] Indice keyframe corrente negativo (" << current_keyframe_index << ")!" << std::endl;
                 return std::make_pair(-1, 0.0f);
            }
            auto new_kf = std::make_shared<KeyFrame>();
            new_kf->index = current_keyframe_index;

             std::pair<int, float> result;
             try {
                result = self.detectLoopClosureID(candidate_kfs, new_kf);
             } catch (const std::exception& e) {
                  std::cerr << "[ERRORE bindings] Eccezione C++ durante detectLoopClosureID: " << e.what() << std::endl;
                  throw; // Rilancia a Python
             } catch (...) {
                  std::cerr << "[ERRORE bindings] Eccezione C++ sconosciuta durante detectLoopClosureID!" << std::endl;
                  throw;
             }
            return result;
        }, "Rileva i loop. Ritorna (id_loop, diff_yaw_rad). id_loop=-1 se non trova nulla.");
}