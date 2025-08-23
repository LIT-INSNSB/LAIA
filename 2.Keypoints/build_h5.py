# -*- coding: utf-8 -*-
# Construye HDF5 de FRAMES con keypoints de manos (42x3) + etiquetas de consenso.
#
# Lee el manifest del Paso 1:
#   LAIA-Net/1.LabelAlignment/Manifests/videos.csv
# columnas:
#   dataset_id,video_id,camera_id,fps,width,height,num_frames,path_video,path_consensus
#
# Uso:
#   python3 build_frames_h5.py \
#       --manifest ../1.LabelAlignment/Manifests/videos.csv \
#       --estimator mediapipe \
#       --out-h5 ../prepared/h5/frames_mediapipe.h5 \
#       --coord-space image \
#       --stride 1
#
# Notas:
# - Importa dinámicamente: keypointEstimators.<estimator> (p.ej. mediapipe_functions)
# - Guarda por video en /videos/<video_id>:
#     data(T,42,3), lw(T,), rw(T,), is_washing(T,), movement(T,), transition(T,)
#   y atributos (dataset_id, camera_id, fps, width, height) en el grupo.



import os
import csv
import cv2
import h5py
import time
import argparse
import importlib
import numpy as np

def _file_exists(p):
    try:
        return os.path.isfile(p)
    except:
        return False

def _dir_ensure(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _load_consensus_csv(path_csv, stride=1):
    """
    Carga consenso y (opcionalmente) submuestrea con 'stride' para alinear
    si también submuestreamos frames.
    Devuelve dict:
      is_washing(int8), movement(int8), transition(uint8)
    """
    if not _file_exists(path_csv):
        return None
    isw, mv, tr = [], [], []
    with open(path_csv, "r") as f:
        reader = csv.DictReader(f)
        idx = 0
        for row in reader:
            if (idx % stride) == 0:
                isw.append(int(row.get("is_washing_cons", -1)))
                mv.append(int(row.get("movement_cons", -1)))
                tr.append(int(row.get("transition_mask", 0)))
            idx += 1
    out = {
        "is_washing": np.array(isw, dtype=np.int8),
        "movement":   np.array(mv,  dtype=np.int8),
        "transition": np.array(tr,  dtype=np.uint8),
    }
    return out

def _open_h5(out_h5):
    if _file_exists(out_h5):
        base, ext = os.path.splitext(out_h5)
        old = base + "_old" + ext
        print("Existe H5. Backup ->", old)
        if _file_exists(old):
            os.remove(old)
        os.replace(out_h5, old)
    _dir_ensure(os.path.dirname(out_h5))
    return h5py.File(out_h5, "w")

def _ensure_meta(h5f, estimator, coord_space):
    if "meta" not in h5f:
        g = h5f.create_group("meta")
        g.attrs["estimator"]   = estimator
        g.attrs["coord_space"] = coord_space
        g.attrs["C"] = 3
        g.attrs["K"] = 42

def _create_video_group(h5f, video_id, attrs):
    root = "videos"
    if root not in h5f:
        h5f.create_group(root)
    if video_id in h5f[root]:
        del h5f[root][video_id]
    g = h5f[root].create_group(video_id)
    for k, v in attrs.items():
        try:
            g.attrs[k] = v
        except:
            pass
    return g

def process_video_row(row, kp_mod, model, h5f, coord_space, stride):
    path_video = row["path_video"]
    path_cons  = row.get("path_consensus", "")
    video_id   = row["video_id"]

    if not _file_exists(path_video):
        print("[WARN] No existe video:", path_video)
        return

    cons = _load_consensus_csv(path_cons, stride=stride) if path_cons else None

    cap = cv2.VideoCapture(path_video)
    if not cap.isOpened():
        print("[WARN] No pude abrir:", path_video)
        return

    n  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    try:
        if abs(fps - round(fps)) < 0.01:
            fps = int(round(fps))
    except:
        pass

    data_list = []
    lw_list   = []
    rw_list   = []

    read_idx = 0
    st = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if (read_idx % stride) == 0:
            kp_42x3, lw, rw = kp_mod.frame_process(model, frame, coord_space=coord_space)
            if not isinstance(kp_42x3, np.ndarray):
                kp_42x3 = np.asarray(kp_42x3, dtype=np.float32)
            if kp_42x3.shape != (42, 3):
                tmp = np.zeros((42, 3), dtype=np.float32)
                try:
                    tmp[:kp_42x3.shape[0], :kp_42x3.shape[1]] = kp_42x3
                except:
                    pass
                kp_42x3 = tmp
            kp_42x3 = kp_42x3.astype(np.float32, copy=False)

            data_list.append(kp_42x3)
            lw_list.append(1 if lw else 0)
            rw_list.append(1 if rw else 0)

        read_idx += 1

    cap.release()

    T = len(data_list)
    if T == 0:
        print("[WARN] Sin frames procesados:", video_id)
        return

    data_np = np.stack(data_list, axis=0)        # (T,42,3)
    lw_np   = np.asarray(lw_list, dtype=np.uint8)
    rw_np   = np.asarray(rw_list, dtype=np.uint8)

    if cons is not None:
        # recorta a longitud mínima por seguridad
        T_cons = cons["is_washing"].shape[0]
        Tm = min(T, T_cons)
        data_np = data_np[:Tm]
        lw_np   = lw_np[:Tm]
        rw_np   = rw_np[:Tm]
        isw = cons["is_washing"][:Tm]
        mov = cons["movement"][:Tm]
        trn = cons["transition"][:Tm]
        T = Tm
    else:
        isw = np.full((T,), -1, dtype=np.int8)
        mov = np.full((T,), -1, dtype=np.int8)
        trn = np.zeros((T,), dtype=np.uint8)

    g = _create_video_group(h5f, video_id, {
        "dataset_id": row.get("dataset_id", ""),
        "camera_id":  row.get("camera_id", ""),
        "fps":        int(fps) if fps is not None else int(row.get("fps", 30)),
        "width":      w,
        "height":     h
    })

    comp = dict(compression="gzip", compression_opts=4)
    g.create_dataset("data", data=data_np, dtype="float32", chunks=(min(120, T), 42, 3), **comp)
    g.create_dataset("lw", data=lw_np, dtype="uint8", chunks=(min(4096, T),), **comp)
    g.create_dataset("rw", data=rw_np, dtype="uint8", chunks=(min(4096, T),), **comp)
    g.create_dataset("is_washing", data=isw, dtype="int8", chunks=(min(4096, T),), **comp)
    g.create_dataset("movement",   data=mov, dtype="int8", chunks=(min(4096, T),), **comp)
    g.create_dataset("transition", data=trn, dtype="uint8", chunks=(min(4096, T),), **comp)

    dt = time.time() - st
    print("[OK] {} | frames:{} | tiempo:{:.2f}s".format(video_id, T, dt))

def main():
    parser = argparse.ArgumentParser(description="Frames HDF5 (hands 42x3) para LAIA")
    default_manifest = os.path.abspath(os.path.join(
        os.path.dirname(__file__), os.pardir, "1.LabelAlignment", "Manifests", "videos.csv"
    ))
    parser.add_argument("--manifest", type=str, default=default_manifest, help="Ruta a videos.csv")
    parser.add_argument("--estimator", type=str, default="mediapipe", help="keypointEstimators.<estimator> (p.ej. mediapipe)")
    parser.add_argument("--out-h5", type=str, required=True, help="Ruta de salida del HDF5 de frames")
    parser.add_argument("--coord-space", type=str, default="image", choices=["image","world"], help="Espacio de coordenadas")
    parser.add_argument("--stride", type=int, default=1, help="Procesar 1 de cada N frames")
    parser.add_argument("--limit", type=int, default=0, help="Procesar solo los primeros N videos (0=todos)")
    args = parser.parse_args()

    if not _file_exists(args.manifest):
        raise RuntimeError("No existe manifest: {}".format(args.manifest))

    # import dinámico del estimador
    if args.estimator.lower() == "mediapipe":
        kp_mod = importlib.import_module("mediapipe_funcs")
    else:
        raise RuntimeError("Estimador no soportado: {}".format(args.estimator))

    # abrir h5 y meta
    h5f = _open_h5(args.out_h5)
    _ensure_meta(h5f, args.estimator, args.coord_space)

    # init modelo (pasando coord_space si aplica)
    try:
        model = kp_mod.model_init(coord_space=args.coord_space)
    except TypeError:
        model = kp_mod.model_init()

    # recorrer manifest
    count = 0
    with open(args.manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if args.limit and count >= args.limit:
                break
            if not _file_exists(row["path_video"]):
                continue
            process_video_row(row, kp_mod, model, h5f, args.coord_space, max(1, args.stride))
            count += 1

    # cerrar
    try:
        kp_mod.close_model(model)
    except:
        pass
    h5f.close()
    print("[DONE] frames HDF5:", args.out_h5)

if __name__ == "__main__":
    main()