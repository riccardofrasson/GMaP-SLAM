# Salva come GTSAM_optimizer_final.py (nella cartella python_src)
#
# OBIETTIVO:
# Questo script unisce il caricamento dati avanzato (Radar+Camera+GPS)
# di 'Detect_Loops_From_RCI_Poses_Camera_GPS.py' con la logica di
# ottimizzazione e calcolo metriche (ATE/RPE) di 'GTSAM_Optimizer.py'.
#
# DIPENDENZE:
# Assicurati che 'loop_detector_frontend.py' sia nella stessa cartella.
#
# BIBLIOTECHE RICHIESTE:
# gtsam, numpy, pandas, matplotlib, scipy, pyproj, torch, torchvision, pillow

import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import math
import warnings

# --- Import GTSAM e SciPy (dal vecchio script) ---
import gtsam
import scipy.linalg  # Per Umeyama/alignment

# --- Import Geodesia (dal nuovo script) ---
try:
    import pyproj

    PYPROJ_AVAILABLE = True
except ImportError:
    print(
        "\nATTENZIONE: Libreria 'pyproj' non trovata. Impossibile usare la conversione GPS ENU."
    )
    print("Installa con: pip install pyproj")
    PYPROJ_AVAILABLE = False

# --- Import Camera / PyTorch (dal nuovo script) ---
try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as transforms
    from PIL import Image, ImageFile

    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    TORCH_AVAILABLE = True
except ImportError:
    print("\nATTENZIONE: PyTorch, Torchvision o PIL non trovati.")
    print("La verifica tramite camera sarà disabilitata.\n")
    TORCH_AVAILABLE = False

# --- Import dal NUOVO Loop Detector ---
try:
    # Importa la classe e la funzione helper dal tuo script
    from loop_detector_frontend import PythonLoopDetector, calculate_pose_difference

    print("Importato PythonLoopDetector da 'loop_detector_frontend.py' (OK).")
except ImportError as e:
    print(
        f"ERRORE: Impossibile importare PythonLoopDetector da loop_detector_frontend: {e}. "
    )
    print("Assicurati che 'loop_detector_frontend.py' sia nella stessa cartella.")
    exit()

# --- Setup EfficientNetV2-S (dal nuovo script) ---
if TORCH_AVAILABLE:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nCaricamento modello EfficientNetV2-S (Torchvision) su {device}...")
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        target_size = (384, 384)
        preprocess = transforms.Compose(
            [
                transforms.Resize(target_size, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=image_mean, std=image_std),
            ]
        )
        feature_extractor = nn.Sequential(
            *list(model.features.children()), nn.AdaptiveAvgPool2d((1, 1))
        )
        feature_extractor.eval().to(device)
        print("Modello EfficientNetV2-S (features + GAP) caricato.")

        def load_and_extract_features(image_path):
            try:
                img = Image.open(image_path).convert("RGB")
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    features_tensor = feature_extractor(img_tensor)
                return features_tensor.squeeze().cpu().numpy()
            except Exception as e:
                return None

        print("Eseguo pre-riscaldamento modello EfficientNet...")
        _ = load_and_extract_features(Image.new("RGB", (640, 480), color="white"))
        print("Pre-riscaldamento OK.")
    except Exception as e:
        print(f"ERRORE CRITICO: Impossibile inizializzare PyTorch/EfficientNet: {e}")
        feature_extractor = None
else:
    feature_extractor = None

if feature_extractor is None:
    TORCH_AVAILABLE = False

    def load_and_extract_features(image_path):
        return None


# --- Fine Setup EfficientNet ---


# --- NUOVA AGGIUNTA: Funzioni di caricamento TUM (da ATE_Test.py) ---
def quaternion_to_rotation_matrix_tum(q):
    x, y, z, w = q
    m = np.empty((3, 3), dtype=np.float64)
    m[0, 0] = 1 - 2 * (y**2 + z**2)
    m[0, 1] = 2 * (x * y - w * z)
    m[0, 2] = 2 * (x * z + w * y)
    m[1, 0] = 2 * (x * y + w * z)
    m[1, 1] = 1 - 2 * (x**2 + z**2)
    m[1, 2] = 2 * (y * z - w * x)
    m[2, 0] = 2 * (x * z - w * y)
    m[2, 1] = 2 * (y * z + w * x)
    m[2, 2] = 1 - 2 * (x**2 + y**2)
    return m


def pose_7d_to_matrix_tum(pose_7d):
    if pose_7d is None or len(pose_7d) != 7:
        return None
    t = pose_7d[:3]
    q_xyzw = pose_7d[3:]
    if (
        np.isnan(t).any()
        or np.isinf(t).any()
        or np.isnan(q_xyzw).any()
        or np.isinf(q_xyzw).any()
    ):
        return None
    q_norm = np.linalg.norm(q_xyzw)
    if q_norm < 1e-6:
        return None
    q_xyzw = q_xyzw / q_norm
    try:
        rot_mat = quaternion_to_rotation_matrix_tum(q_xyzw)
    except Exception as e:
        print(f"Errore conv quat manuale: {q_xyzw} -> {e}")
        return None
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = t
    return T.astype(np.float64)


def load_trajectory_tum_baseline(filepath):
    timestamps = []
    poses_mat = []
    line_count = 0
    errors = 0
    try:
        with open(filepath, "r") as f:
            for line in f:
                line_count += 1
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 8:
                    errors += 1
                    print(
                        f"Attenzione (BASELINE): Riga {line_count} ignorata (formato)."
                    )
                    continue
                try:
                    ts = float(parts[0])  # Timestamp
                    pose_7d = np.array([float(p) for p in parts[1:8]])
                    # Usa il loader specifico del file TUM
                    T_mat = pose_7d_to_matrix_tum(pose_7d)
                    if T_mat is not None:
                        timestamps.append(ts)
                        poses_mat.append(T_mat)
                    else:
                        errors += 1
                        print(
                            f"Attenzione (BASELINE): Riga {line_count} ignorata (conv.)."
                        )
                except ValueError:
                    errors += 1
                    print(
                        f"Attenzione (BASELINE): Riga {line_count} ignorata (float err)."
                    )
    except FileNotFoundError:
        print(f"ERRORE (BASELINE): File non trovato: {filepath}")
        return None, None
    except Exception as e:
        print(f"ERRORE (BASELINE): Imprevisto: {e}")
        return None, None
    if errors > 0:
        print(f"Caricamento Baseline con {errors} righe ignorate.")
    if len(poses_mat) < 2:
        print(f"ERRORE (BASELINE): Troppo corta ({len(poses_mat)}).")
        return None, None
    print(f"Caricate {len(poses_mat)} pose Baseline valide.")
    # --- MODIFICA: Ritorna solo le pose, gli indici saranno usati dal file sync ---
    return poses_mat


# --- FINE NUOVA AGGIUNTA ---


def read_csv_trajectory(filepath, is_liosam=False):
    data = []
    with open(filepath, "r") as f:
        is_new_format = False
        for line in f:
            if line.startswith("%time"):
                is_new_format = True
                continue
            if line.strip() == "":
                continue
            parts = line.strip().split(",")

            # VINS output format (vio.csv / VINS_RURAL_B0.csv): t, x, y, z, qw, qx, qy, qz...
            if len(parts) == 11 or len(parts) == 12:
                try:
                    t = float(parts[0]) * 1e-9
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    qw, qx, qy, qz = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )
                    data.append((t, x, y, z, qx, qy, qz, qw))
                except:
                    pass
            # ROS bag exported CSV (LIO-SAM, 4DRadarSLAM)
            elif len(parts) > 12:
                try:
                    t = float(parts[0]) * 1e-9
                    x, y, z = float(parts[5]), float(parts[6]), float(parts[7])
                    qx, qy, qz, qw = (
                        float(parts[8]),
                        float(parts[9]),
                        float(parts[10]),
                        float(parts[11]),
                    )
                    data.append((t, x, y, z, qx, qy, qz, qw))
                except:
                    pass
    return data


# --- Funzioni Helper (dal nuovo script 'Loop_Detector.py') ---


def load_and_filter_radar_raw(filepath):
    """
    Carica, TRASFORMA nel frame della posa (X_front, Y_lat, Z_alt),
    e filtra i dati radar.
    """
    try:
        radar_data_raw = np.load(filepath, allow_pickle=True)
        if (
            radar_data_raw.size == 0
            or radar_data_raw.ndim < 2
            or radar_data_raw.shape[1] < 3
        ):
            return np.zeros((0, 4), dtype=np.float32)

        if radar_data_raw.shape[1] >= 5:
            intensity_col = radar_data_raw[:, 4:5]
        else:
            intensity_col = np.zeros((radar_data_raw.shape[0], 1))
        radar_data = np.zeros((radar_data_raw.shape[0], 4))
        radar_data[:, 0] = radar_data_raw[:, 2]  # X_new (Front) = Z_old (Front, idx 2)
        radar_data[:, 1] = radar_data_raw[:, 0]  # Y_new (Lat)   = X_old (Lat, idx 0)
        radar_data[:, 2] = radar_data_raw[:, 1]  # Z_new (Alt)   = Y_old (Alt, idx 1)
        radar_data[:, 3] = intensity_col[:, 0]  # Intensity (idx 3)

        finite_mask = np.isfinite(radar_data[:, :3]).all(axis=1)
        origin_mask = np.any(radar_data[:, :3] != 0, axis=1)
        valid_points_mask = finite_mask & origin_mask
        radar_data = radar_data[valid_points_mask]
        if radar_data.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        distances = np.linalg.norm(radar_data[:, :3], axis=1)
        radar_data = radar_data[distances < 400.0]
        if radar_data.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        z_coords_altezza = radar_data[:, 2]
        radar_data = radar_data[(z_coords_altezza > -50.0) & (z_coords_altezza < 3.0)]
        if radar_data.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        x_coords_profondita = radar_data[:, 0]
        radar_data = radar_data[x_coords_profondita > 0]
        if radar_data.shape[0] == 0:
            return np.zeros((0, 4), dtype=np.float32)

        return radar_data.astype(np.float32)

    except FileNotFoundError:
        return None
    except Exception as e:
        print(
            f"Errore imprevisto durante caricamento/filtro radar {filepath.name}: {e}"
        )
        return None


def pose_7d_to_matrix(pose_7d):
    """
    Converte posa [tx,ty,tz, qx,qy,qz,qw] in matrice 4x4.
    (Questa è la versione del MAIN SCRIPT, usa SciPy)
    """
    if pose_7d is None or len(pose_7d) != 7:
        return None
    t = pose_7d[:3]
    q_xyzw = pose_7d[3:]
    if (
        np.isnan(t).any()
        or np.isinf(t).any()
        or np.isnan(q_xyzw).any()
        or np.isinf(q_xyzw).any()
    ):
        return None
    q_norm = np.linalg.norm(q_xyzw)
    if q_norm < 1e-6:
        return None
    q_xyzw = q_xyzw / q_norm
    try:
        rot_mat = R.from_quat(q_xyzw).as_matrix()
    except ValueError as e:
        return None
    T = np.eye(4)
    T[:3, :3] = rot_mat
    T[:3, 3] = t
    return T.astype(np.float32)


def reconstruct_gt_trajectory(gt_relative_df, indices_df, gt_lidar_col_name):
    """Ricostruisce la traiettoria GT assoluta dalle pose relative GT."""
    print("Ricostruzione traiettoria GT...")
    gt_absolute_poses_mat = [np.eye(4, dtype=np.float32)]
    gt_relative_poses_mat_valid = []
    num_sync_frames = len(indices_df)
    max_allowable_lidar_index_gt = len(gt_relative_df)
    current_gt_pose = np.eye(4, dtype=np.float32)
    processed_valid_steps = 0

    if gt_lidar_col_name not in indices_df.columns:
        print(
            f"ERRORE CRITICO: La colonna GT '{gt_lidar_col_name}' non esiste in indices_df per la ricostruzione GT."
        )
        return [], []

    for i in tqdm(range(num_sync_frames), desc="  Ricostruendo GT", leave=False):
        try:
            row_i = indices_df.iloc[i]
            lidar_idx_gt = int(row_i[gt_lidar_col_name])

            if lidar_idx_gt < max_allowable_lidar_index_gt:
                gt_rel_pose_row = gt_relative_df.iloc[lidar_idx_gt]
                gt_rel_pose_7d = np.array(
                    [
                        gt_rel_pose_row[k]
                        for k in ["dx", "dy", "dz", "qx", "qy", "qz", "qw"]
                    ]
                )
                # Usa il pose_7d_to_matrix standard (scipy)
                T_rel_gt = pose_7d_to_matrix(gt_rel_pose_7d)
                if T_rel_gt is None:
                    raise ValueError(f"Pose GT relativa invalida {lidar_idx_gt}")
                T_rel_gt = T_rel_gt.astype(np.float32)
                gt_relative_poses_mat_valid.append(T_rel_gt.copy())
                current_gt_pose = current_gt_pose @ T_rel_gt
                gt_absolute_poses_mat.append(current_gt_pose.copy())
                processed_valid_steps += 1
            else:
                break
        except (KeyError, ValueError, IndexError, TypeError) as e:
            break
    num_abs_poses = processed_valid_steps + 1
    gt_absolute_poses_mat = gt_absolute_poses_mat[:num_abs_poses]
    return gt_absolute_poses_mat, gt_relative_poses_mat_valid


def load_and_convert_gps_enu(gps_filepath):
    """
    Carica il file GPS.txt, lo converte in coordinate ENU locali usando pyproj
    e ritorna un DataFrame indicizzato con 'gps_east' e 'gps_north'.
    """
    if not PYPROJ_AVAILABLE:
        print("  ERRORE: pyproj non disponibile per la conversione ENU.")
        return None

    print(f"Caricamento e conversione GPS (ENU) da {gps_filepath}...")
    gps_data_raw = []
    required_cols = 6  # index, ts_sec, ts_nsec, lat, lon, alt
    line_errors = 0
    try:
        with open(gps_filepath, "r") as f:
            for line_num, line in enumerate(f):
                try:
                    if not isinstance(line, str):
                        line_errors += 1
                        continue

                    tag_index = line.find("'")
                    if tag_index != -1:
                        data_part = line[:tag_index]
                    else:
                        data_part = line

                    cleaned_data = data_part.strip()
                    if not cleaned_data:
                        continue

                    parts = cleaned_data.split()
                    if len(parts) >= required_cols:
                        idx = int(
                            parts[0]
                        )  # Questo è il 'gps_index' (allineato a lidar_index)
                        lat = float(parts[3])
                        lon = float(parts[4])
                        alt = float(parts[5])
                        if -90 <= lat <= 90 and -180 <= lon <= 180:
                            gps_data_raw.append(
                                {"index": idx, "lat": lat, "lon": lon, "alt": alt}
                            )
                        else:
                            line_errors += 1
                    else:
                        line_errors += 1
                except Exception as line_exc:
                    line_errors += 1
                    continue

        if line_errors > 0:
            print(
                f"  Attenzione: {line_errors} righe saltate durante il parsing del file GPS."
            )
        if not gps_data_raw:
            print("  Nessun dato GPS valido trovato nel file dopo il parsing.")
            return None

        gps_df_raw = (
            pd.DataFrame(gps_data_raw)
            .drop_duplicates(subset="index")
            .set_index("index")
        )
        if gps_df_raw.empty:
            print("  DataFrame GPS vuoto dopo rimozione duplicati.")
            return None

        ref_lat = gps_df_raw["lat"].iloc[0]
        ref_lon = gps_df_raw["lon"].iloc[0]
        ref_alt = gps_df_raw["alt"].iloc[0]
        print(
            f"  Riferimento ENU: Lat {ref_lat:.6f}, Lon {ref_lon:.6f}, Alt {ref_alt:.2f}"
        )

        crs_wgs84 = pyproj.CRS("EPSG:4326")
        crs_ecef = pyproj.CRS("EPSG:4978")
        transformer_to_ecef = pyproj.Transformer.from_crs(
            crs_wgs84, crs_ecef, always_xy=True
        )

        ecef_x, ecef_y, ecef_z = transformer_to_ecef.transform(
            gps_df_raw["lon"].values, gps_df_raw["lat"].values, gps_df_raw["alt"].values
        )
        ref_ecef_x, ref_ecef_y, ref_ecef_z = transformer_to_ecef.transform(
            ref_lon, ref_lat, ref_alt
        )

        lon_rad = math.radians(ref_lon)
        lat_rad = math.radians(ref_lat)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        dx = ecef_x - ref_ecef_x
        dy = ecef_y - ref_ecef_y
        dz = ecef_z - ref_ecef_z
        gps_east = -sin_lon * dx + cos_lon * dy
        gps_north = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz

        gps_df_raw["gps_east"] = gps_east
        gps_df_raw["gps_north"] = gps_north

        print(f"  Conversione GPS in ENU completata ({len(gps_df_raw)} punti validi).")
        return gps_df_raw[["gps_east", "gps_north"]]

    except FileNotFoundError:
        print(f"  ERRORE: File GPS non trovato.")
        return None
    except Exception as e:
        print(f"  ERRORE CRITICO durante caricamento/conversione GPS ENU: {e}")
        return None


# --- FINE FUNZIONI HELPER (dal nuovo script) ---


# --- Funzioni GTSAM e Metriche (dal vecchio script 'GTSAM_Optimizer.py') ---


def build_and_optimize_graph(pred_abs_poses, pred_rel_poses, verified_loops):
    """
    Costruisce un grafo di pose in GTSAM con RUMORE DINAMICO (basato su binning),
    lo ottimizza e ritorna le pose ottimizzate.
    """
    print("\n" + "=" * 50)
    print("Inizio Costruzione Grafo GTSAM (con Rumore Dinamico basato su Binning)...")

    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    # Definiamo i livelli di fiducia. Valori bassi = alta fiducia (molla rigida)
    STRAIGHT_SIGMAS = np.array([0.005, 0.005, 0.005, 0.05, 0.05, 0.05])
    straight_noise = gtsam.noiseModel.Diagonal.Sigmas(STRAIGHT_SIGMAS)
    LIGHT_TURN_SIGMAS = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    light_turn_noise = gtsam.noiseModel.Diagonal.Sigmas(LIGHT_TURN_SIGMAS)
    SHARP_TURN_SIGMAS = np.array([0.05, 0.05, 0.05, 0.2, 0.2, 0.2])
    sharp_turn_noise = gtsam.noiseModel.Diagonal.Sigmas(SHARP_TURN_SIGMAS)
    BINS_THRESHOLDS = {"straight": 2.0, "light_turn": 10.0, "sharp_turn": float("inf")}
    dt = 0.1  # ASSUNZIONE: l'odometria è a 10Hz
    print(
        f"  - Logica Incertezza: Rettilineo (<{BINS_THRESHOLDS['straight']} deg/s), Curva Leggera (<{BINS_THRESHOLDS['light_turn']} deg/s), Curva Stretta"
    )
    LOOP_SIGMAS = np.array([0.5, 0.5, 0.5, 0.05, 0.05, 0.05])
    base_loop_noise = gtsam.noiseModel.Diagonal.Sigmas(LOOP_SIGMAS)
    loop_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(1.0), base_loop_noise
    )
    PRIOR_SIGMAS = np.array([1e-6] * 6)
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(PRIOR_SIGMAS)

    first_pose_mat = pred_abs_poses[0]
    first_pose_gtsam = gtsam.Pose3(first_pose_mat)
    graph.add(gtsam.PriorFactorPose3(0, first_pose_gtsam, prior_noise))
    initial_estimates.insert(0, first_pose_gtsam)
    print(f"Aggiunto Prior (Ancora) al Nodo 0.")

    num_odom_factors = len(pred_rel_poses)
    bin_counts = {key: 0 for key in BINS_THRESHOLDS}

    for i in range(num_odom_factors):
        T_rel_mat = pred_rel_poses[i]
        T_rel_gtsam = gtsam.Pose3(T_rel_mat)

        _, rot_rad = calculate_pose_difference(T_rel_mat, np.eye(4))
        rate_deg_s = (rot_rad / dt) * (180.0 / np.pi)

        if rate_deg_s < BINS_THRESHOLDS["straight"]:
            noise_to_use = straight_noise
            bin_counts["straight"] += 1
        elif rate_deg_s < BINS_THRESHOLDS["light_turn"]:
            noise_to_use = light_turn_noise
            bin_counts["light_turn"] += 1
        else:
            noise_to_use = sharp_turn_noise
            bin_counts["sharp_turn"] += 1

        graph.add(gtsam.BetweenFactorPose3(i, i + 1, T_rel_gtsam, noise_to_use))
        T_abs_mat = pred_abs_poses[i + 1]
        initial_estimates.insert(i + 1, gtsam.Pose3(T_abs_mat))

    print(f"Aggiunti {num_odom_factors} fattori odometria:")
    print(f"    - Rettilinei: {bin_counts['straight']}")
    print(f"    - Curve Leggere: {bin_counts['light_turn']}")
    print(f"    - Curve Strette: {bin_counts['sharp_turn']}")

    num_loops = 0
    # NOTA: Il nuovo loop detector ritorna (idx_curr, idx_loop, T_icp, fitness, cam_flag)
    # Prendiamo solo i primi 3 elementi che servono a GTSAM.
    for loop_data in verified_loops:
        idx_curr, idx_loop, T_icp_loop_curr = loop_data[0], loop_data[1], loop_data[2]

        if idx_curr >= len(pred_abs_poses) or idx_loop >= len(pred_abs_poses):
            print(
                f"Attenzione: Loop {idx_curr}->{idx_loop} scartato (indici fuori range)."
            )
            continue

        T_loop_curr_gtsam = gtsam.Pose3(T_icp_loop_curr)
        graph.add(
            gtsam.BetweenFactorPose3(idx_loop, idx_curr, T_loop_curr_gtsam, loop_noise)
        )
        num_loops += 1

    print(f"Aggiunti {num_loops} fattori di loop closure.")
    if num_loops == 0:
        print("Nessun loop trovato. L'ottimizzazione non produrrà cambiamenti.")
        # Ritorna le pose originali se non ci sono loop
        return pred_abs_poses

    print("Ottimizzazione del grafo in corso (Levenberg-Marquardt)...")
    try:
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("SUMMARY")
        params.setAbsoluteErrorTol(1e-8)
        params.setRelativeErrorTol(1e-8)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        result = optimizer.optimize()
        print("Ottimizzazione completata!")
    except Exception as e:
        print(f"ERRORE CRITICO during ottimizzazione GTSAM: {e}")
        return pred_abs_poses

    optimized_poses_mat = []
    for i in range(len(pred_abs_poses)):
        try:
            pose_gtsam = result.atPose3(i)
            optimized_poses_mat.append(pose_gtsam.matrix())
        except Exception as e:
            print(f"Attenzione: Impossibile estrarre posa {i}. Uso stima iniziale.")
            optimized_poses_mat.append(pred_abs_poses[i])

    print("Pose ottimizzate estratte.")
    print("=" * 50 + "\n")
    return optimized_poses_mat


def align_umeyama(model, data):
    """Calcola la trasformazione di allineamento (Sim(3)) da data a model."""
    model_mean = model.mean(axis=0)
    data_mean = data.mean(axis=0)
    model_centered = model - model_mean
    data_centered = data - data_mean
    cov_matrix = data_centered.T @ model_centered / len(model)
    U, S, Vt = scipy.linalg.svd(cov_matrix)
    V = Vt.T
    det_UVt = np.linalg.det(U @ Vt)
    diag_fix = np.diag([1] * (model.shape[1] - 1) + [det_UVt])
    R_align = V @ diag_fix @ U.T
    var_data = np.var(data_centered, axis=0).sum()
    c = np.trace(np.diag(S) @ diag_fix) / var_data if var_data > 1e-8 else 1.0
    t = model_mean - c * R_align @ data_mean
    T = np.eye(4)
    T[:3, :3] = c * R_align
    T[:3, 3] = t
    return T, t, R_align, c


# --- MODIFICA: Riportato a com'era prima. Ritorna solo ATE ---
def calculate_ate(gt_poses_mat, estimated_poses_mat):
    """
    Calcola l'Absolute Trajectory Error (ATE) RMSE dopo allineamento Umeyama.
    RITORNA: solo ate_rmse
    """
    # Converti in lista se è un array numpy per il check
    if isinstance(gt_poses_mat, np.ndarray):
        gt_poses_mat = list(gt_poses_mat)
    if isinstance(estimated_poses_mat, np.ndarray):
        estimated_poses_mat = list(estimated_poses_mat)

    if len(gt_poses_mat) != len(estimated_poses_mat) or len(gt_poses_mat) < 2:
        print(
            f"Errore ATE: Le traiettorie non hanno la stessa lunghezza ({len(gt_poses_mat)} vs {len(estimated_poses_mat)}) o sono troppo corte."
        )
        return None

    gt_positions = np.array([pose[:3, 3] for pose in gt_poses_mat])
    est_positions = np.array([pose[:3, 3] for pose in estimated_poses_mat])

    try:
        T_align, _, _, _ = align_umeyama(gt_positions, est_positions)
    except Exception as e:
        print(f"Errore durante align_umeyama in calculate_ate: {e}")
        return None  # Ritorna None

    est_positions_hom = np.hstack((est_positions, np.ones((len(est_positions), 1))))
    est_positions_aligned_hom = (T_align @ est_positions_hom.T).T
    est_positions_aligned = est_positions_aligned_hom[:, :3]
    translation_errors = np.linalg.norm(gt_positions - est_positions_aligned, axis=1)
    ate_rmse = np.sqrt(np.mean(translation_errors**2))

    # Ritorna solo l'errore
    return ate_rmse


def calculate_kitti_rpe(
    gt_poses, est_poses, subsequence_lengths=[100, 200, 300, 400, 500, 600, 700, 800]
):
    """
    Calcola le metriche RPE stile KITTI (drift traslazionale % e rotazionale deg/m)
    usando il metodo ufficiale per il calcolo della distanza percorsa.
    """
    if len(gt_poses) != len(est_poses) or len(gt_poses) < 2:
        print("Errore RPE: Traiettorie di lunghezza diversa o troppo corte.")
        return None

    num_poses = len(gt_poses)
    results = {}

    total_trans_errors = {L: [] for L in subsequence_lengths}
    total_rot_errors_rad = {L: [] for L in subsequence_lengths}

    print(
        f"Calcolo RPE (Metodo Ufficiale KITTI) su {num_poses} pose per lunghezze {subsequence_lengths}m..."
    )

    gt_rel_poses = []
    for k in range(1, num_poses):
        try:
            gt_rel = np.linalg.inv(gt_poses[k - 1].astype(np.float64)) @ gt_poses[
                k
            ].astype(np.float64)
            gt_rel_poses.append(gt_rel)
        except np.linalg.LinAlgError:
            print(
                f"Attenzione RPE: Errore inversione GT pose {k-1}. Impossibile calcolare RPE."
            )
            return None

    for i in tqdm(range(num_poses), desc="  Calcolando RPE", leave=False):
        T_gt_i = gt_poses[i].astype(np.float64)
        T_est_i = est_poses[i].astype(np.float64)
        current_distance = 0.0
        for j in range(i + 1, num_poses):
            segment_dist = np.linalg.norm(gt_rel_poses[j - 1][:3, 3])
            current_distance += segment_dist
            T_gt_j = gt_poses[j].astype(np.float64)
            T_est_j = est_poses[j].astype(np.float64)

            for length in subsequence_lengths:
                if current_distance >= length and len(total_trans_errors[length]) == i:
                    try:
                        T_gt_rel_ij = np.linalg.inv(T_gt_i) @ T_gt_j
                        T_est_rel_ij = np.linalg.inv(T_est_i) @ T_est_j
                        T_error_ij = np.linalg.inv(T_est_rel_ij) @ T_gt_rel_ij
                        trans_error = np.linalg.norm(T_error_ij[:3, 3])
                        _, rot_error_rad = calculate_pose_difference(
                            T_error_ij, np.eye(4)
                        )
                        total_trans_errors[length].append(trans_error)
                        total_rot_errors_rad[length].append(rot_error_rad)
                    except np.linalg.LinAlgError:
                        total_trans_errors[length].append(np.nan)
                        total_rot_errors_rad[length].append(np.nan)

            all_found_for_i = all(
                len(total_trans_errors[L]) > i for L in subsequence_lengths
            )
            if all_found_for_i:
                break

    for length in subsequence_lengths:
        trans_errors_np = np.array(total_trans_errors[length])
        rot_errors_rad_np = np.array(total_rot_errors_rad[length])
        valid_mask = ~np.isnan(trans_errors_np) & ~np.isnan(rot_errors_rad_np)
        valid_trans_errors = trans_errors_np[valid_mask]
        valid_rot_errors_rad = rot_errors_rad_np[valid_mask]
        count = len(valid_trans_errors)

        if count > 0:
            avg_trans_err = np.mean(valid_trans_errors)
            avg_rot_err_rad = np.mean(valid_rot_errors_rad)
            trans_drift_percent = (avg_trans_err / length) * 100
            rot_drift_deg_per_m = np.degrees(avg_rot_err_rad) / length
            results[length] = {
                "count": count,
                "avg_trans_err_m": avg_trans_err,
                "avg_rot_err_deg": np.degrees(avg_rot_err_rad),
                "trans_drift_percent": trans_drift_percent,
                "rot_drift_deg_per_m": rot_drift_deg_per_m,
            }
        else:
            results[length] = None
    return results


# --- FINE FUNZIONI GTSAM E METRICHE ---


# --- Script Principale (Basato su 'Loop_Detector.py' e integrato con GTSAM) ---
if __name__ == "__main__":
    # 1. PARSING ARGOMENTI (dal nuovo script, più completo)
    parser = argparse.ArgumentParser(
        description="Esegue Loop Closure (Radar+Cam+GPS) E Ottimizzazione GTSAM."
    )
    # Argomenti comuni
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/media/arrubuntu20/SSD_2/Riccardo/Extracted Dataset",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="/media/arrubuntu20/SSD_2/Riccardo/Codici/loop_closure_module/python_src/evaluation_results_RCI_FINAL-Paper",
    )
    parser.add_argument("--sequence_name", type=str, required=True)
    parser.add_argument("--save_plot", action="store_true")

    # --- NUOVA AGGIUNTA: Argomenti per baseline CSV ---
    parser.add_argument(
        "--liosam_csv", type=str, default=None, help="Percorso del CSV LIO-SAM."
    )
    parser.add_argument(
        "--vins_csv", type=str, default=None, help="Percorso del CSV VINS-Fusion."
    )
    parser.add_argument(
        "--fdr_csv", type=str, default=None, help="Percorso del CSV 4DRadarSLAM."
    )
    parser.add_argument(
        "--bag_start_time",
        type=float,
        default=None,
        help="Timestamp di inizio della bag (rosbag info).",
    )

    # Argomenti Loop Detector
    parser.add_argument("--sc_thresh", type=float, default=0.6)
    parser.add_argument(
        "--lc_dist_thresh",
        type=float,
        default=310.0,
        help="Raggio ricerca Odom (fallback).",
    )
    parser.add_argument("--min_frames_interval", type=int, default=300)
    parser.add_argument("--icp_fitness", type=float, default=0.5)
    parser.add_argument(
        "--icp_max_dist", type=float, default=0.5
    )  # N.B. La classe interna usa GICP, questo è per Odom check
    parser.add_argument("--odom_err_trans", type=float, default=0.1)
    parser.add_argument("--odom_err_rot", type=float, default=0.03)
    parser.add_argument("--stationary_trans", type=float, default=0.05)
    parser.add_argument("--stationary_rot", type=float, default=0.5)
    parser.add_argument("--post_stationary_cooldown", type=int, default=100)

    # Argomenti Camera
    parser.add_argument("--cam_sim_thresh", type=float, default=0.92)

    # Argomenti GPS
    parser.add_argument(
        "--use_gps_filter",
        action="store_true",
        help="Usa il GPS per il pre-filtering spaziale (RACCOMANDATO).",
    )
    parser.add_argument(
        "--gps_search_radius",
        type=float,
        default=15.0,
        help="Raggio di ricerca (in metri) per il filtro GPS.",
    )
    parser.add_argument(
        "--gps_file_name",
        type=str,
        default="GPS.txt",
        help="Nome del file GPS dentro la cartella della sequenza.",
    )

    args = parser.parse_args()

    # --- NUOVA AGGIUNTA: Controllo argomenti baseline CSV ---
    if (
        args.liosam_csv or args.vins_csv or args.fdr_csv
    ) and args.bag_start_time is None:
        print(
            "ERRORE: Per usare i CSV dei modelli è obbligatorio specificare --bag_start_time."
        )
        exit(1)

    # Controllo PyTorch e Pyproj
    if not TORCH_AVAILABLE:
        args.cam_sim_thresh = 1.0
    elif feature_extractor is None:
        args.cam_sim_thresh = 1.0
    if not PYPROJ_AVAILABLE and args.use_gps_filter:
        print("Pyproj non disponibile, filtro GPS disabilitato.")
        args.use_gps_filter = False

    # 2. INIZIALIZZAZIONE LOOP DETECTOR (dal nuovo script)
    print("\nInizializzazione Loop Detector (da loop_detector_frontend.py)...")
    try:
        loop_detector = PythonLoopDetector(
            sc_dist_thresh=args.sc_thresh,
            accum_dist_thresh=args.lc_dist_thresh,
            icp_fitness_thresh=args.icp_fitness,
            icp_max_correspondence=args.icp_max_dist,  # N.B. GICP interno usa valori diversi
            odom_check_trans_thresh_m_per_m=args.odom_err_trans,
            odom_check_rot_thresh_rad_per_m=args.odom_err_rot,
            stationary_trans_thresh_m=args.stationary_trans,
            stationary_rot_thresh_deg=args.stationary_rot,
            post_stationary_cooldown_frames=args.post_stationary_cooldown,
            cam_sim_thresh=args.cam_sim_thresh,
            use_gps_filter=args.use_gps_filter,
            gps_search_radius_m=args.gps_search_radius,
        )
        loop_detector.MIN_FRAMES_INTERVAL = args.min_frames_interval
        print(
            f"  Loop Detector OK (SC: {args.sc_thresh}, Odom Fallback: {args.lc_dist_thresh}m, Int: {args.min_frames_interval} frames)"
        )
        print(
            f"  Verifica: ICP Fit>={args.icp_fitness}, Odom Trans<={args.odom_err_trans}m/m, Odom Rot<={args.odom_err_rot}rad/m"
        )
        print(
            f"  Filtro Stazionario: Trans<{args.stationary_trans:.3f}m AND Rot<{args.stationary_rot:.2f}deg."
        )
        print(f"  Cooldown Post-Stazionario: {args.post_stationary_cooldown} frames.")
    except Exception as e:
        print(f"Errore inizializzazione Loop Detector: {e}")
        exit()

    # 3. CARICAMENTO E PREPARAZIONE DATI (dal nuovo script)
    print("\nCaricamento e preparazione dati (inclusi GPS e Camera)...")
    sequence_dir = Path(args.data_dir) / args.sequence_name
    indices_path = sequence_dir / "synchronized_indices.csv"
    pred_absolute_poses_path = (
        Path(args.results_dir) / f"predicted_poses_7d_{args.sequence_name}.npy"
    )
    gt_relative_path = sequence_dir / "relative_poses.csv"
    gps_path = sequence_dir / args.gps_file_name

    missing_files = []
    if not sequence_dir.exists():
        missing_files.append(str(sequence_dir))
    if not indices_path.exists():
        missing_files.append(str(indices_path))
    if not pred_absolute_poses_path.exists():
        missing_files.append(str(pred_absolute_poses_path))
    if not gt_relative_path.exists():
        missing_files.append(str(gt_relative_path))

    if missing_files:
        print("ERRORE CRITICO: I seguenti file o directory sono mancanti:")
        for f in missing_files:
            print(f"  - {f}")
        exit()

    try:
        indices_df = pd.read_csv(indices_path)
        pred_absolute_poses_7d = np.load(pred_absolute_poses_path)
        gt_relative_df = pd.read_csv(gt_relative_path)
    except Exception as e:
        print(f"Errore caricamento dati base: {e}")
        exit()

    # Carica e Converti GPS usando ENU
    gps_enu_df = None
    if args.use_gps_filter:
        gps_enu_df = load_and_convert_gps_enu(gps_path)
        if gps_enu_df is None:
            print(
                "Disabilito filtro GPS a causa di errore caricamento/conversione ENU."
            )
            args.use_gps_filter = False
            loop_detector.USE_GPS_FILTER = False

    # Validazione e Preparazione Pose Predette
    pred_absolute_poses_mat = []
    valid_original_indices_pred = []
    for i in range(pred_absolute_poses_7d.shape[0]):
        # Usa il pose_7d_to_matrix standard (scipy)
        T_abs_pred = pose_7d_to_matrix(pred_absolute_poses_7d[i])
        if T_abs_pred is not None:
            pred_absolute_poses_mat.append(T_abs_pred)
            valid_original_indices_pred.append(i)
    num_valid_pred_poses = len(pred_absolute_poses_mat)
    if num_valid_pred_poses < 2:
        print("Errore: Pose predette valide insufficienti.")
        exit()

    try:
        max_valid_index = (
            max(valid_original_indices_pred) if valid_original_indices_pred else -1
        )
        if len(indices_df) <= max_valid_index:
            raise IndexError(
                f"Errore: File indici ({len(indices_df)} righe) più corto di max pose idx ({max_valid_index})."
            )
        indices_df_filtered = indices_df.iloc[valid_original_indices_pred].reset_index(
            drop=True
        )
        if len(indices_df_filtered) != num_valid_pred_poses:
            min_len_post_filter = min(num_valid_pred_poses, len(indices_df_filtered))
            pred_absolute_poses_mat = pred_absolute_poses_mat[:min_len_post_filter]
            indices_df_filtered = indices_df_filtered.iloc[:min_len_post_filter]
            num_valid_pred_poses = min_len_post_filter
    except IndexError as e:
        print(f"Errore fatale filtraggio indices_df: {e}.")
        exit()

    # Ricostruisci Traiettoria GT
    gt_lidar_col_name = (
        "lidar_index" if "lidar_index" in indices_df_filtered.columns else "lidar_index"
    )  # Default robusto
    gt_absolute_poses_mat, _ = reconstruct_gt_trajectory(
        gt_relative_df, indices_df_filtered, gt_lidar_col_name
    )

    # Recupera Percorsi Radar, Camera E GPS ENU
    print("Recupero percorsi Radar, Camera e dati GPS ENU...")
    radar_filepaths_ordered = []
    camera_filepaths_ordered = []
    gps_enu_ordered = []

    rad_idx_col = (
        "radar_index"
        if "radar_index" in indices_df_filtered.columns
        else "camera_radar_index"
    )
    lid_idx_col = gt_lidar_col_name

    is_cam_enabled = TORCH_AVAILABLE
    cam_idx_col = None
    if is_cam_enabled:
        if "camera_index" in indices_df_filtered.columns:
            cam_idx_col = "camera_index"
        elif "camera_radar_index" in indices_df_filtered.columns:
            cam_idx_col = "camera_radar_index"
        else:
            is_cam_enabled = False
    camera_dir = sequence_dir / "image_left"
    if is_cam_enabled and not camera_dir.exists():
        is_cam_enabled = False

    print(
        f"  Indici -> Radar:'{rad_idx_col}', Cam:'{cam_idx_col}'(Abil:{is_cam_enabled}), GPS Lookup:'{lid_idx_col}'(Abil:{args.use_gps_filter})"
    )

    for i in range(len(indices_df_filtered)):
        radar_path, cam_path, gps_enu = None, None, None
        try:
            row = indices_df_filtered.iloc[i]
            radar_idx = int(row[rad_idx_col])
            lidar_idx_for_gps = int(row[lid_idx_col])
            radar_path = sequence_dir / "radar" / f"{radar_idx:06d}.npy"
            if not radar_path.exists():
                radar_path = None
            if is_cam_enabled:
                cam_idx = int(row[cam_idx_col])
                cam_path = camera_dir / f"{cam_idx:06d}.png"
                if not cam_path.exists():
                    cam_path = camera_dir / f"{cam_idx:06d}.jpg"
                if not cam_path.exists():
                    cam_path = None
            if args.use_gps_filter and gps_enu_df is not None:
                try:
                    gps_row = gps_enu_df.loc[lidar_idx_for_gps]
                    gps_enu = gps_row[["gps_east", "gps_north"]].values.astype(
                        np.float32
                    )
                    if np.isnan(gps_enu).any():
                        gps_enu = None
                except KeyError:
                    gps_enu = None
                except Exception:
                    gps_enu = None
        except (KeyError, ValueError, IndexError, TypeError):
            pass
        radar_filepaths_ordered.append(radar_path)
        camera_filepaths_ordered.append(cam_path)
        gps_enu_ordered.append(gps_enu)

    # Allinea Liste
    lengths = [
        len(pred_absolute_poses_mat),
        len(radar_filepaths_ordered),
        len(camera_filepaths_ordered),
        len(gps_enu_ordered),
    ]
    final_num_frames = min(lengths)
    gt_eval_len = min(final_num_frames, len(gt_absolute_poses_mat))

    print(
        f"Allineamento finale a {final_num_frames} frames. GT per valutazione: {gt_eval_len} frames."
    )

    # Applica il taglio finale per coerenza
    pred_absolute_poses_mat = pred_absolute_poses_mat[:gt_eval_len]
    gt_absolute_poses_mat = gt_absolute_poses_mat[:gt_eval_len]
    radar_filepaths_ordered = radar_filepaths_ordered[:gt_eval_len]
    camera_filepaths_ordered = camera_filepaths_ordered[:gt_eval_len]
    gps_enu_ordered = gps_enu_ordered[:gt_eval_len]

    final_num_frames = gt_eval_len  # Aggiorna il numero finale
    num_rel_expected = final_num_frames - 1 if final_num_frames > 0 else 0

    # Ricalcola Relative Predette (sulla base dei dati finali allineati)
    pred_relative_poses_mat_valid = []
    if final_num_frames > 0:
        for i in range(1, final_num_frames):
            try:
                T_i = pred_absolute_poses_mat[i]
                T_i_minus_1 = pred_absolute_poses_mat[i - 1]
                T_rel = np.linalg.inv(T_i_minus_1) @ T_i
                if np.isnan(T_rel).any() or np.isinf(T_rel).any():
                    raise np.linalg.LinAlgError
                pred_relative_poses_mat_valid.append(T_rel.astype(np.float32))
            except np.linalg.LinAlgError:
                pred_relative_poses_mat_valid.append(np.eye(4, dtype=np.float32))

    # Verifica Finale Lunghezze
    if not (
        len(pred_absolute_poses_mat)
        == len(gt_absolute_poses_mat)
        == len(radar_filepaths_ordered)
        == len(camera_filepaths_ordered)
        == len(gps_enu_ordered)
        == final_num_frames
        and len(pred_relative_poses_mat_valid) == num_rel_expected
        and final_num_frames > 0
    ):
        print(
            "\nERRORE INTERNO CRITICO: Incoerenza lunghezze DOPO ALLINEAMENTO FINALE!"
        )
        print(f"  Frames Finali Attesi: {final_num_frames}")
        print(
            f"  Pred: {len(pred_absolute_poses_mat)}, GT: {len(gt_absolute_poses_mat)}, Radar: {len(radar_filepaths_ordered)}"
        )
        print(
            f"  Cam: {len(camera_filepaths_ordered)}, GPS: {len(gps_enu_ordered)}, Rel Pred: {len(pred_relative_poses_mat_valid)} (attese {num_rel_expected})"
        )
        exit()
    else:
        print(
            f"Caricamento e preparazione dati completati ({final_num_frames} frames)."
        )

    # 4. FASE DI RILEVAMENTO LOOP (dal nuovo script)
    print(
        f"\nInizio processamento sequenza '{args.sequence_name}' ({final_num_frames} frames)..."
    )
    keyframe_count = 0

    for i in tqdm(range(final_num_frames), desc="Processando Frames", leave=True):
        radar_path = radar_filepaths_ordered[i]
        cam_path = camera_filepaths_ordered[i]
        current_abs_pose_mat = pred_absolute_poses_mat[i]
        relative_odom_np = pred_relative_poses_mat_valid[i - 1] if i > 0 else None
        current_gps_enu = gps_enu_ordered[i]

        cam_features = None
        if is_cam_enabled and cam_path is not None:
            cam_features = load_and_extract_features(cam_path)

        if radar_path is None:
            continue
        raw_radar_cloud_np = load_and_filter_radar_raw(radar_path)

        if raw_radar_cloud_np is not None and raw_radar_cloud_np.shape[0] > 0:
            if raw_radar_cloud_np.shape[1] == 4:
                try:
                    loop_detector.add_keyframe(
                        raw_radar_cloud_np,
                        current_abs_pose_mat,
                        relative_odom_np,
                        cam_features,
                        current_gps_enu,
                    )
                    keyframe_count += 1
                except Exception as e:
                    print(f"\nErrore add_keyframe frame {i}: {e}")

    print(
        f"\nProcessamento sequenza completato. Aggiunti {keyframe_count} keyframe validi."
    )

    # Stampa Statistiche GPS
    print(f"\n--- Statistiche Filtro GPS (Debug) ---")
    total_checks = (
        loop_detector.gps_filter_successes + loop_detector.gps_filter_fallbacks
    )
    if total_checks > 0:
        perc_success = (loop_detector.gps_filter_successes / total_checks) * 100
        perc_fallback = (loop_detector.gps_filter_fallbacks / total_checks) * 100
        print(f"  Coppie totali analizzate dal filtro spaziale: {total_checks}")
        print(
            f"  Tentativi GPS (dati ENU presenti): {loop_detector.gps_filter_successes} ({perc_success:.1f}%)"
        )
        print(
            f"  Fallback Odom (dati ENU mancanti): {loop_detector.gps_filter_fallbacks} ({perc_fallback:.1f}%)"
        )
    else:
        print("  Nessuna coppia di candidati analizzata dal filtro spaziale.")

    verified_loops = loop_detector.get_detected_loops()
    print(f"\nRilevati e VERIFICATI {len(verified_loops)} loop closures.")

    # Stampa riepilogo loop trovati
    if verified_loops:
        print("Elenco Loop Verificati (current_idx -> loop_idx [Tipo]):")
        for i, (
            idx_curr,
            idx_loop,
            T_icp,
            gicp_fitness,
            cam_override_flag,
        ) in enumerate(verified_loops):
            is_super_loop_approx = cam_override_flag and (
                gicp_fitness >= loop_detector.ICP_FITNESS_THRESH
            )
            if is_super_loop_approx:
                loop_type_str = "[SUPER]"
            elif cam_override_flag:
                loop_type_str = "[CAM]"
            else:
                loop_type_str = "[RADAR]"
            print(
                f"  - Loop {i+1}: {idx_curr} -> {idx_loop} {loop_type_str} (Fit:{gicp_fitness:.3f})"
            )

    # 5. FASE DI OTTIMIZZAZIONE GTSAM (dal vecchio script)
    #    (Chiamata alla funzione definita sopra)
    optimized_poses = build_and_optimize_graph(
        pred_absolute_poses_mat, pred_relative_poses_mat_valid, verified_loops
    )

    # --- NUOVA AGGIUNTA: Caricamento, Sync e ATE CSV ---
    gt_final_synced = []
    opt_final_synced = []
    liosam_final_synced = []
    vins_final_synced = []
    fdr_final_synced = []

    ate_opt_synced = None
    ate_liosam_synced = None
    ate_vins_synced = None
    ate_fdr_synced = None

    has_baselines = args.liosam_csv or args.vins_csv or args.fdr_csv

    if has_baselines and args.bag_start_time is not None:
        print("\nCaricamento Traiettorie CSV Baseline...")
        liosam_data = (
            read_csv_trajectory(args.liosam_csv, is_liosam=True)
            if args.liosam_csv
            else []
        )
        vins_data = read_csv_trajectory(args.vins_csv) if args.vins_csv else []
        fdr_data = read_csv_trajectory(args.fdr_csv) if args.fdr_csv else []

        print("Esecuzione sincronizzazione a 5 vie (GT, SLAM, LIO-SAM, VINS, 4DR)...")
        for i in range(final_num_frames):
            gt_idx_original = int(indices_df_filtered.iloc[i][gt_lidar_col_name])
            gt_timestamp = args.bag_start_time + gt_idx_original * 0.1

            def find_match(data, target_t):
                if not data:
                    return None
                times = np.array([d[0] for d in data])
                idx = np.argmin(np.abs(times - target_t))
                if np.abs(times[idx] - target_t) < 0.2:
                    if len(data[idx]) == 8:
                        t, x, y, z, qx, qy, qz, qw = data[idx]
                        mat = np.eye(4, dtype=np.float32)
                        mat[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
                        mat[0, 3], mat[1, 3], mat[2, 3] = x, y, z
                        return mat
                return None

            l_mat = find_match(liosam_data, gt_timestamp) if liosam_data else None
            v_mat = find_match(vins_data, gt_timestamp) if vins_data else None
            f_mat = find_match(fdr_data, gt_timestamp) if fdr_data else None

            gt_final_synced.append(gt_absolute_poses_mat[i])
            opt_final_synced.append(optimized_poses[i])
            liosam_final_synced.append(l_mat)
            vins_final_synced.append(v_mat)
            fdr_final_synced.append(f_mat)

        print(f"Sincronizzazione completata su {len(gt_final_synced)} frame GT.")

        def compute_baseline_ate(gt_list, base_list):
            valid_gt = []
            valid_base = []
            for g, b in zip(gt_list, base_list):
                if b is not None:
                    valid_gt.append(g)
                    valid_base.append(b)
            if len(valid_gt) < 2:
                return None, valid_base
            return calculate_ate(valid_gt, valid_base), valid_base

        ate_opt_synced = calculate_ate(gt_final_synced, opt_final_synced)

        if liosam_data:
            ate_liosam_synced, _ = compute_baseline_ate(
                gt_final_synced, liosam_final_synced
            )
        if vins_data:
            ate_vins_synced, _ = compute_baseline_ate(
                gt_final_synced, vins_final_synced
            )
        if fdr_data:
            ate_fdr_synced, _ = compute_baseline_ate(gt_final_synced, fdr_final_synced)

    # 6. CALCOLO METRICHE (ATE + RPE) (dal vecchio script)
    print("\n" + "=" * 50)
    print("Calcolo Metriche di Valutazione...")

    # Calcolo ATE (Sulle traiettorie complete)
    ate_pred = calculate_ate(gt_absolute_poses_mat, pred_absolute_poses_mat)
    ate_opt = calculate_ate(gt_absolute_poses_mat, optimized_poses)

    # Calcolo RPE (Sulle traiettorie complete)
    rpe_pred_results = calculate_kitti_rpe(
        gt_absolute_poses_mat, pred_absolute_poses_mat
    )
    rpe_opt_results = calculate_kitti_rpe(gt_absolute_poses_mat, optimized_poses)

    # Allineamento Hand-Eye globale (SVD) per calcolo full RPE (Traslazione + Rotazione)
    def align_and_compute_full_rpe(gt_list, base_list):
        import scipy.linalg

        valid_gt, valid_base = [], []
        for g, b in zip(gt_list, base_list):
            if b is not None:
                valid_gt.append(g)
                valid_base.append(b)
        if len(valid_gt) < 2:
            return None

        # 1. Estrai le posizioni per l'allineamento della mappa
        gt_pos = np.array([p[:3, 3] for p in valid_gt])
        base_pos = np.array([p[:3, 3] for p in valid_base])

        # 2. Allineamento globale (Map Frame) tramite Umeyama
        model_mean = gt_pos.mean(axis=0)
        data_mean = base_pos.mean(axis=0)
        model_centered = gt_pos - model_mean
        data_centered = base_pos - data_mean
        cov_matrix = data_centered.T @ model_centered / len(gt_pos)
        U, S, Vt = scipy.linalg.svd(cov_matrix)
        det_UVt = np.linalg.det(U @ Vt.T)
        diag_fix = np.diag([1, 1, np.sign(det_UVt)])
        R_align_map = Vt.T @ diag_fix @ U.T
        t_align_map = model_mean - R_align_map @ data_mean

        # 3. Calcolo ottimale degli Estrinseci del Sensore (R_ext) tramite SVD su tutta la traiettoria
        # Vogliamo R_ext tale che: R_align_map * R_base * R_ext ≈ R_gt
        # R_ext ≈ (R_align_map * R_base)^T * R_gt
        M = np.zeros((3, 3))
        for g, b in zip(valid_gt, valid_base):
            R_gt = g[:3, :3]
            R_base = b[:3, :3]
            # Accumula la matrice per SVD
            M += (R_align_map @ R_base).T @ R_gt

        U_ext, S_ext, Vt_ext = scipy.linalg.svd(M)
        det_ext = np.linalg.det(U_ext @ Vt_ext)
        diag_ext = np.diag([1, 1, np.sign(det_ext)])
        R_ext_optimal = U_ext @ diag_ext @ Vt_ext

        # 4. Applica Map Alignment e Sensor Extrinsics a tutte le pose
        fully_aligned_poses = []
        for b in valid_base:
            # Crea la trasformazione globale
            T_map = np.eye(4)
            T_map[:3, :3] = R_align_map
            T_map[:3, 3] = t_align_map

            # Crea la trasformazione estrinseca del sensore
            T_ext = np.eye(4)
            T_ext[:3, :3] = R_ext_optimal

            # P_corrected = T_map * P_base * T_ext
            P_corrected = T_map @ b @ T_ext
            fully_aligned_poses.append(P_corrected)

        # 5. Ora le pose calcolate hanno assi e posizioni identiche alla GT.
        # Possiamo usare la funzione ufficiale KITTI per RPE!
        return calculate_kitti_rpe(valid_gt, fully_aligned_poses)

    rpe_liosam_results = align_and_compute_full_rpe(
        gt_final_synced, liosam_final_synced
    )
    rpe_vins_results = align_and_compute_full_rpe(gt_final_synced, vins_final_synced)
    rpe_fdr_results = align_and_compute_full_rpe(gt_final_synced, fdr_final_synced)

    # Stampa ATE
    print("\n--- Absolute Trajectory Error (ATE RMSE) ---")
    print(
        f"  (Calcolato su tutti i {final_num_frames} frame disponibili per GT/Pred/Opt)"
    )
    if ate_pred is not None:
        print(f"  - Predetta (Input) vs GT:   {ate_pred:.4f} metri")
    else:
        print("  - Predetta (Input) vs GT:   N/A")
    if ate_opt is not None:
        print(f"  - Ottimizzata (SLAM) vs GT: {ate_opt:.4f} metri")
    else:
        print("  - Ottimizzata (SLAM) vs GT: N/A")

    if ate_pred is not None and ate_opt is not None:
        improvement = ate_pred - ate_opt
        improvement_percent = (improvement / ate_pred) * 100 if ate_pred > 1e-6 else 0
        print(
            f"  - Miglioramento (SLAM vs Predetta):  {improvement:.4f} metri ({improvement_percent:.2f}%)"
        )

    # --- NUOVA AGGIUNTA: Stampa ATE Sincronizzato ---
    if has_baselines and ate_opt_synced is not None:
        print("\n  --- ATE Sincronizzato (per confronto Baseline) ---")
        print(f"  (Calcolato su {len(gt_final_synced)} frame sincronizzati)")
        if ate_liosam_synced is not None:
            print(f"  - LIO-SAM vs GT:          {ate_liosam_synced:.4f} metri")
        if ate_vins_synced is not None:
            print(f"  - VINS-Fusion vs GT:      {ate_vins_synced:.4f} metri")
        if ate_fdr_synced is not None:
            print(f"  - 4DRadarSLAM vs GT:      {ate_fdr_synced:.4f} metri")
        print(f"  - Ottimizzata (SLAM) vs GT: {ate_opt_synced:.4f} metri\n")

    # Stampa RPE (Ora con tutti i modelli)
    print("\n--- Relative Pose Error (RPE - KITTI Metrics) ---")
    print(f"  (Calcolato su tutti i frame sincronizzati disponibili)")

    # Formattazione colonna
    col_w = 11
    print(
        f"{'Len (m)':<7} | {'Ours (Opt)':<12} | {'Ours (Pred)':<12} | {'LIO-SAM':<12} | {'VINS':<12} | {'4DRadar':<12}"
    )
    print("-" * 80)

    kitti_lengths_to_print = [100, 200, 300, 400, 500, 600, 700, 800]

    print(">>> TRANSLATION DRIFT (%)")
    for length in kitti_lengths_to_print:
        opt_res = rpe_opt_results.get(length) if rpe_opt_results else None
        pred_res = rpe_pred_results.get(length) if rpe_pred_results else None
        lio_res = rpe_liosam_results.get(length) if rpe_liosam_results else None
        vins_res = rpe_vins_results.get(length) if rpe_vins_results else None
        fdr_res = rpe_fdr_results.get(length) if rpe_fdr_results else None

        opt_t = f"{opt_res['trans_drift_percent']:.2f}" if opt_res else "-"
        pred_t = f"{pred_res['trans_drift_percent']:.2f}" if pred_res else "-"
        lio_t = f"{lio_res['trans_drift_percent']:.2f}" if lio_res else "-"
        vins_t = f"{vins_res['trans_drift_percent']:.2f}" if vins_res else "-"
        fdr_t = f"{fdr_res['trans_drift_percent']:.2f}" if fdr_res else "-"

        # Stampa solo se almeno un modello ha dati per questa lunghezza
        if any(
            res is not None for res in [opt_res, pred_res, lio_res, vins_res, fdr_res]
        ):
            print(
                f"{length:<7} | {opt_t:<12} | {pred_t:<12} | {lio_t:<12} | {vins_t:<12} | {fdr_t:<12}"
            )

    print("\n>>> ROTATION DRIFT (deg/m)")
    for length in kitti_lengths_to_print:
        opt_res = rpe_opt_results.get(length) if rpe_opt_results else None
        pred_res = rpe_pred_results.get(length) if rpe_pred_results else None
        lio_res = rpe_liosam_results.get(length) if rpe_liosam_results else None
        vins_res = rpe_vins_results.get(length) if rpe_vins_results else None
        fdr_res = rpe_fdr_results.get(length) if rpe_fdr_results else None

        opt_r = f"{opt_res['rot_drift_deg_per_m']:.4f}" if opt_res else "-"
        pred_r = f"{pred_res['rot_drift_deg_per_m']:.4f}" if pred_res else "-"
        lio_r = f"{lio_res['rot_drift_deg_per_m']:.4f}" if lio_res else "-"
        vins_r = f"{vins_res['rot_drift_deg_per_m']:.4f}" if vins_res else "-"
        fdr_r = f"{fdr_res['rot_drift_deg_per_m']:.4f}" if fdr_res else "-"

        if any(
            res is not None for res in [opt_res, pred_res, lio_res, vins_res, fdr_res]
        ):
            print(
                f"{length:<7} | {opt_r:<12} | {pred_r:<12} | {lio_r:<12} | {vins_r:<12} | {fdr_r:<12}"
            )

    print("=" * 80 + "\n")

    # 7. FASE DI VISUALIZZAZIONE (Rivista per estetica "Tesi" e 4 Plot)
    print("Visualizzazione risultati (versione 'Tesi' su 3+1 plot)...")

    # --- MODIFICA: Prendi le traiettorie XY grezze (non allineate) ---
    gt_trajectory_xy = np.array([pose[0:2, 3] for pose in gt_absolute_poses_mat])
    pred_trajectory_xy = np.array([pose[0:2, 3] for pose in pred_absolute_poses_mat])
    opt_trajectory_xy = np.array([pose[0:2, 3] for pose in optimized_poses])

    if gt_trajectory_xy.shape[0] < 1:
        print("Errore: Liste pose vuote, impossibile plottare.")
        exit()

    # --- Setup comune per tutti i plot ---

    # Creazione directory di salvataggio
    save_dir = Path("PaperAblationSpatialCamera")
    save_dir.mkdir(exist_ok=True, parents=True)

    # Impostazioni font professionali (Dimensioni aumentate)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 18,  # Aumentato da 14
            "axes.labelsize": 22,  # Aumentato da 16
            "xtick.labelsize": 16,  # Aumentato da 12
            "ytick.labelsize": 16,  # Aumentato da 12
            "legend.fontsize": 18,  # Aumentato da 12
            "axes.titlesize": 24,  # Aumentato da 18
            "pdf.fonttype": 42,  # FORZA FONT TYPE 42 (TRUETYPE) INVECE DEL TIPO 3 (VIETATO DA IEEE)
            "ps.fonttype": 42,
        }
    )

    # --- MODIFICA Colori ---
    # Palette colori uniformata ai grafici di comparison
    gt_color = "black"  # Ground Truth massicciamente visibile
    pred_color = "#00CED1"  # Turchese scuro/Ciano (nuovo colore mai usato)
    opt_color = "#FF4500"  # Arancione scuro ultra-visibile (SLAM)
    baseline_color = "#9467bd"  # Viola Tab10
    loop_cmap = plt.cm.get_cmap("tab10", 10)

    # Etichette con ATE (per i primi 3 plot) - Ora a DUE cifre decimali
    gt_label = "Ground Truth"
    pred_label = (
        f"Predicted (ATE: {ate_pred:.2f}m)"
        if ate_pred is not None
        else "Predicted (Input)"
    )
    opt_label = (
        f"Optimized (ATE: {ate_opt:.2f}m)"
        if ate_opt is not None
        else "Optimized (GTSAM)"
    )

    axis_xlabel = "East [m]" if args.use_gps_filter else "X [m]"
    axis_ylabel = "North [m]" if args.use_gps_filter else "Y [m]"

    # Logica Loop (per plot 2, 3 e 4)
    def plot_loops(verified_loops, trajectory_xy, max_loops_in_legend=7):
        loop_handles_map = {}
        if verified_loops:
            for i, (idx_curr, idx_loop, T_icp, _, cam_flag) in enumerate(
                verified_loops
            ):
                if idx_curr < len(trajectory_xy) and idx_loop < len(trajectory_xy):
                    loop_color = loop_cmap(i % 10)
                    line_style = (
                        "--" if cam_flag else "-"
                    )  # CAM = tratteggiato, RADAR = solido
                    # loop_type_str = "[CAM]" if cam_flag else "[RADAR]"
                    point_curr = trajectory_xy[idx_curr]
                    point_loop = trajectory_xy[idx_loop]
                    loop_label = (
                        f"Loop {i+1} ({idx_loop} -> {idx_curr})"  # {loop_type_str}'
                    )

                    (line,) = plt.plot(
                        [point_curr[0], point_loop[0]],
                        [point_curr[1], point_loop[1]],
                        linestyle=line_style,
                        color=loop_color,
                        alpha=0.8,
                        linewidth=1.5,
                        zorder=4,
                        label=(loop_label if i < max_loops_in_legend else None),
                    )

                    scatter_start = plt.scatter(
                        point_loop[0],
                        point_loop[1],
                        color=loop_color,
                        marker="x",
                        s=60,
                        linewidths=2.0,
                        zorder=5,
                    )

                    scatter_end = plt.scatter(
                        point_curr[0],
                        point_curr[1],
                        facecolors=loop_color,
                        marker="o",
                        s=60,
                        edgecolors="black",
                        zorder=5,
                    )

                    if i < max_loops_in_legend:
                        loop_handles_map[loop_label] = (
                            line,
                            scatter_start,
                            scatter_end,
                        )
        return loop_handles_map

    # Funzione helper per legenda INTERNA con loc='best'
    def create_legend(ax, handles_map, ordered_labels, title=None, loc="best"):
        handles_for_legend, labels_for_legend = [], []
        for label in ordered_labels:
            if label in handles_map:
                handles_for_legend.append(handles_map[label])
                labels_for_legend.append(label)

        if handles_for_legend:
            ax.legend(
                handles_for_legend,
                labels_for_legend,
                loc=loc,  # 'best'
                title=title,
                fancybox=True,  # Aggiunge bordi arrotondati
                framealpha=0.8,  # Leggermente trasparente
            )

    # --- FINE SETUP COMUNE ---

    # --- PLOT 1: PREDETTA vs. GT ---
    print("Generazione Plot 1: Predetta vs. GT...")
    fig1, ax1 = plt.subplots(figsize=(12, 10))

    handles_map_1 = {}
    if gt_trajectory_xy.shape[0] > 0:
        (gt_line,) = ax1.plot(
            gt_trajectory_xy[:, 0],
            gt_trajectory_xy[:, 1],
            color=gt_color,
            linestyle="--",
            label=gt_label,
            linewidth=4.0,
            alpha=1.0,
            zorder=2,
        )
        handles_map_1[gt_label] = gt_line

    (pred_line,) = ax1.plot(
        pred_trajectory_xy[:, 0],
        pred_trajectory_xy[:, 1],
        color=pred_color,
        linestyle="--",
        label=pred_label,
        linewidth=4.0,
        alpha=1.0,
        zorder=1,
    )
    handles_map_1[pred_label] = pred_line

    ax1.set_title(f"Predicted Trajectory vs. Ground Truth - {args.sequence_name}")
    ax1.set_xlabel(axis_xlabel)
    ax1.set_ylabel(axis_ylabel)
    create_legend(ax1, handles_map_1, ordered_labels=[gt_label, pred_label])
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.axis("equal")
    fig1.tight_layout()

    if args.save_plot:
        try:
            save_path_pdf = save_dir / f"{args.sequence_name}_1_pred_gt.pdf"
            save_path_png = save_dir / f"{args.sequence_name}_1_pred_gt.png"
            fig1.savefig(save_path_pdf, bbox_inches="tight")
            fig1.savefig(save_path_png, bbox_inches="tight", dpi=300)
            print(f"  -> Plot 1 salvato in '{save_dir.name}/'")
        except Exception as e:
            print(f"Errore salvataggio plot 1: {e}")
    plt.close(fig1)

    # --- PLOT 2: OTTIMIZZATA (SLAM) vs. GT ---
    print("Generazione Plot 2: Ottimizzata (SLAM) vs. GT...")
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    handles_map_2 = {}
    if gt_trajectory_xy.shape[0] > 0:
        (gt_line,) = ax2.plot(
            gt_trajectory_xy[:, 0],
            gt_trajectory_xy[:, 1],
            color=gt_color,
            linestyle="--",
            label=gt_label,
            linewidth=4.0,
            alpha=1.0,
            zorder=2,
        )
        handles_map_2[gt_label] = gt_line

    (opt_line,) = ax2.plot(
        opt_trajectory_xy[:, 0],
        opt_trajectory_xy[:, 1],
        color=opt_color,
        linestyle="-",
        label=opt_label,
        linewidth=4.0,
        alpha=1.0,
        zorder=3,
    )
    handles_map_2[opt_label] = opt_line

    # Aggiungi loop (passando le coordinate corrette)
    loop_handles = plot_loops(verified_loops, opt_trajectory_xy[:, :2])
    for label, handle_group in loop_handles.items():
        handles_map_2[label] = handle_group[0]  # Aggiungi solo la linea

    ax2.set_title(
        f"Optimized Trajectory (SLAM) vs. Ground Truth - {args.sequence_name}"
    )
    ax2.set_xlabel(axis_xlabel)
    ax2.set_ylabel(axis_ylabel)
    ordered_labels_2 = [gt_label, opt_label]  # Rimosso list(loop_handles.keys())
    create_legend(ax2, handles_map_2, ordered_labels=ordered_labels_2)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.axis("equal")
    fig2.tight_layout()

    if args.save_plot:
        try:
            save_path_pdf = save_dir / f"{args.sequence_name}_2_opt_gt.pdf"
            save_path_png = save_dir / f"{args.sequence_name}_2_opt_gt.png"
            fig2.savefig(save_path_pdf, bbox_inches="tight")
            fig2.savefig(save_path_png, bbox_inches="tight", dpi=300)
            print(f"  -> Plot 2 salvato in '{save_dir.name}/'")
        except Exception as e:
            print(f"Errore salvataggio plot 2: {e}")
    plt.close(fig2)

    # --- PLOT 3: CONFRONTO COMPLETO (Il tuo modello) ---
    print("Generazione Plot 3: Confronto Completo (Input vs. SLAM)...")
    fig3, ax3 = plt.subplots(figsize=(12, 10))

    handles_map_3 = {}
    if gt_trajectory_xy.shape[0] > 0:
        (gt_line,) = ax3.plot(
            gt_trajectory_xy[:, 0],
            gt_trajectory_xy[:, 1],
            color=gt_color,
            linestyle="--",
            label=gt_label,
            linewidth=4.0,
            alpha=1.0,
            zorder=2,
        )
        handles_map_3[gt_label] = gt_line

    (pred_line,) = ax3.plot(
        pred_trajectory_xy[:, 0],
        pred_trajectory_xy[:, 1],
        color=pred_color,
        linestyle="--",
        label=pred_label,
        linewidth=4.0,
        alpha=1.0,
        zorder=1,
    )
    handles_map_3[pred_label] = pred_line

    (opt_line,) = ax3.plot(
        opt_trajectory_xy[:, 0],
        opt_trajectory_xy[:, 1],
        color=opt_color,
        linestyle="-",
        label=opt_label,
        linewidth=4.0,
        alpha=1.0,
        zorder=3,
    )
    handles_map_3[opt_label] = opt_line

    # Aggiungi loop (passando le coordinate corrette)
    loop_handles_3 = plot_loops(verified_loops, opt_trajectory_xy[:, :2])
    for label, handle_group in loop_handles_3.items():
        handles_map_3[label] = handle_group[0]

    ax3.set_title(f"Full Trajectory Comparison - {args.sequence_name}")
    ax3.set_xlabel(axis_xlabel)
    ax3.set_ylabel(axis_ylabel)
    ordered_labels_3 = [gt_label, opt_label, pred_label]
    create_legend(ax3, handles_map_3, ordered_labels=ordered_labels_3)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.axis("equal")
    fig3.tight_layout()

    if args.save_plot:
        try:
            save_path_pdf = save_dir / f"{args.sequence_name}_3_all.pdf"
            save_path_png = save_dir / f"{args.sequence_name}_3_all.png"
            fig3.savefig(save_path_pdf, bbox_inches="tight")
            fig3.savefig(save_path_png, bbox_inches="tight", dpi=300)
            print(f"  -> Plot 3 salvato in '{save_dir.name}/'")
        except Exception as e:
            print(f"Errore salvataggio plot 3: {e}")
    plt.close(fig3)

    # --- NUOVA AGGIUNTA: PLOT 4 (Tutte le Baseline vs. SLAM vs. GT) ---
    if has_baselines and len(gt_final_synced) > 0:
        print("Generazione Plot 4: Confronto Baseline (su frame sincronizzati)...")
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        handles_map_4 = {}

        gt_sync_xy = np.array([pose[0:2, 3] for pose in gt_final_synced])
        opt_sync_xy = np.array([pose[0:2, 3] for pose in opt_final_synced])

        opt_label_synced = f"Ours (ATE: {ate_opt:.2f}m)"

        # Plot GT: Resa MASSICCIAMENTE visibile (Nera, spessa)
        (gt_line,) = ax4.plot(
            gt_sync_xy[:, 0],
            gt_sync_xy[:, 1],
            color="black",
            linestyle="--",
            label="Ground Truth",
            linewidth=4.0,
            alpha=1.0,
            zorder=6,
        )
        handles_map_4["Ground Truth"] = gt_line

        # Plot Ottimizzato: Arancione scuro ultra-visibile
        (opt_line,) = ax4.plot(
            opt_sync_xy[:, 0],
            opt_sync_xy[:, 1],
            color="#FF4500",
            linestyle="-",
            label=opt_label_synced,
            linewidth=4.0,
            alpha=1.0,
            zorder=7,
        )
        handles_map_4[opt_label_synced] = opt_line

        ordered_labels_4 = ["Ground Truth", opt_label_synced]

        def process_and_plot_baseline(base_list, color, style, name, ate_val):
            if ate_val is None:
                return

            valid_base_xy = []
            valid_gt_xy = []
            for g, b in zip(gt_final_synced, base_list):
                if b is not None:
                    valid_base_xy.append(b[0:2, 3])
                    valid_gt_xy.append(g[0:2, 3])

            if len(valid_base_xy) < 2:
                return
            valid_base_xy = np.array(valid_base_xy)
            valid_gt_xy = np.array(valid_gt_xy)

            base_centered = valid_base_xy - valid_base_xy[0]
            gt_centered = valid_gt_xy - valid_gt_xy[0]

            if name == "4DRadarSLAM":
                baseline_flipped_xy = base_centered.copy()
                baseline_flipped_xy[:, 1] = -baseline_flipped_xy[:, 1]
                aligned_base_xy = baseline_flipped_xy + valid_gt_xy[0]
            else:
                idx = min(20, len(base_centered) - 1)
                v_base = base_centered[idx]
                v_gt = gt_centered[idx]

                angle_base = np.arctan2(v_base[1], v_base[0])
                angle_gt = np.arctan2(v_gt[1], v_gt[0])
                theta = angle_gt - angle_base

                c, s = np.cos(theta), np.sin(theta)
                R = np.array([[c, -s], [s, c]])

                base_rotated = (R @ base_centered.T).T
                aligned_base_xy = base_rotated + valid_gt_xy[0]

            lbl = f"{name} (ATE: {ate_val:.2f}m)"
            # Baseline con colori accesi e spessore aumentato
            (line,) = ax4.plot(
                aligned_base_xy[:, 0],
                aligned_base_xy[:, 1],
                color=color,
                linestyle=style,
                label=lbl,
                linewidth=4.0,
                alpha=1.0,
                zorder=4,
            )
            handles_map_4[lbl] = line
            ordered_labels_4.append(lbl)

        # Colori super d'impatto e vivaci
        process_and_plot_baseline(
            liosam_final_synced, "blue", "--", "LIO-SAM", ate_liosam_synced
        )
        process_and_plot_baseline(
            vins_final_synced, "#FF00FF", ":", "VINS-Fusion", ate_vins_synced
        )
        process_and_plot_baseline(
            fdr_final_synced, "#00C900", "-.", "4DRadarSLAM", ate_fdr_synced
        )

        ax4.set_title(f"Baselines vs. Optimized SLAM - {args.sequence_name}")
        ax4.set_xlabel(axis_xlabel)
        ax4.set_ylabel(axis_ylabel)

        create_legend(ax4, handles_map_4, ordered_labels=ordered_labels_4)
        ax4.grid(True, linestyle=":", alpha=0.6)
        ax4.axis("equal")
        fig4.tight_layout()

        if args.save_plot:
            try:
                save_path_pdf = save_dir / f"{args.sequence_name}_4_baseline_comp.pdf"
                save_path_png = save_dir / f"{args.sequence_name}_4_baseline_comp.png"
                fig4.savefig(save_path_pdf, bbox_inches="tight")
                fig4.savefig(save_path_png, bbox_inches="tight", dpi=300)
                print(f"  -> Plot 4 salvato in '{save_dir.name}/'")
            except Exception as e:
                print(f"Errore salvataggio plot 4: {e}")
        plt.close(fig4)
    elif args.bag_start_time is None and has_baselines:
        print(
            "ATTENZIONE: Plot 4 (Baseline) saltato a causa di mancanza di bag_start_time."
        )

    # --- FINE NUOVA AGGIUNTA ---

    if not args.save_plot:
        print("\n--save_plot non specificato. Mostro l'ultimo grafico...")
        # --- MODIFICA: Mostra il plot 4 se esiste, altrimenti il 3 ---
        if "fig4" in locals() and plt.fignum_exists(fig4.number):
            plt.figure(fig4.number)
            plt.show()
        elif "fig3" in locals() and plt.fignum_exists(fig3.number):
            plt.figure(fig3.number)
            plt.show()
        elif "fig2" in locals() and plt.fignum_exists(fig2.number):
            plt.figure(fig2.number)
            plt.show()
        elif "fig1" in locals() and plt.fignum_exists(fig1.number):
            plt.figure(fig1.number)
            plt.show()

    print("\nScript terminato.")
