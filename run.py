# run.py
import os, sys, csv, time, argparse, importlib.util, shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_start_method
from tqdm import tqdm
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
KP_DIR = ROOT / "2.Keypoints"
PARTS_DIR = ROOT / "prepared" / "h5" / "parts"

def _load_by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Globals por proceso
_kp_mod = None
_bh5 = None
_coord_space = "image"
_estimator = "mediapipe"
_stride = 1

def _init_worker(coord_space, estimator, stride):
    # Evitar sobre-suscripción de hilos en cada proceso
    os.environ.setdefault("OMP_NUM_THREADS","1")
    os.environ.setdefault("MKL_NUM_THREADS","1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS","1")

    global _kp_mod, _bh5, _coord_space, _estimator, _stride
    _bh5 = _load_by_path("build_h5", KP_DIR / "build_h5.py")
    if estimator.lower() == "mediapipe":
        _kp_mod = _load_by_path("mediapipe_funcs", KP_DIR / "mediapipe_funcs.py")
    else:
        raise RuntimeError(f"Estimador no soportado: {estimator}")
    _coord_space = coord_space
    _estimator = estimator
    _stride = stride
    PARTS_DIR.mkdir(parents=True, exist_ok=True)

def _safe_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)

def _process_one_video(row: dict):
    """Procesa un video y escribe un H5 parcial en prepared/h5/parts."""
    dataset_id = row.get("dataset_id", "unknown")
    video_id = row["video_id"]
    base = f"{_safe_name(dataset_id)}__{_safe_name(video_id)}.h5"
    out = PARTS_DIR / base

    # Abrir H5 parcial
    h5f = h5py.File(str(out), "w")
    _bh5._ensure_meta(h5f, _estimator, _coord_space)

    # Crear modelo por proceso
    try:
        model = _kp_mod.model_init(coord_space=_coord_space)
    except TypeError:
        model = _kp_mod.model_init()

    ok = True
    err = None
    try:
        # Reutilizamos la función de escritura por video
        _bh5.process_video_row(row, _kp_mod, model, h5f, _coord_space, max(1, _stride))
    except Exception as e:
        ok = False
        err = str(e)
    finally:
        try:
            _kp_mod.close_model(model)
        except Exception:
            pass
        h5f.close()

    return {"ok": ok, "err": err, "dataset_id": dataset_id, "video_id": video_id, "part": str(out)}

def _open_h5_with_backup(path: Path):
    # Equivalente a _open_h5 pero aquí por cuenta propia
    if path.exists():
        base = path.with_suffix("")
        bak = Path(f"{base}_old{path.suffix}")
        print("Existe H5. Backup ->", bak)
        if bak.exists():
            bak.unlink()
        path.replace(bak)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w")

def _ensure_meta(fout, estimator, coord_space):
    if "meta" not in fout:
        g = fout.create_group("meta")
        g.attrs["estimator"] = estimator
        g.attrs["coord_space"] = coord_space
        g.attrs["C"] = 3
        g.attrs["K"] = 42

def _merge_parts(parts_paths, out_h5, estimator, coord_space, show_bar=True):
    """Fusión secuencial segura a un único H5."""
    fout = _open_h5_with_backup(Path(out_h5))
    _ensure_meta(fout, estimator, coord_space)
    if "videos" not in fout:
        fout.create_group("videos")

    it = parts_paths
    if show_bar:
        it = tqdm(parts_paths, desc="Merge", unit="file")

    for p in it:
        with h5py.File(p, "r") as fin:
            if "videos" not in fin:
                continue
            for vid in fin["videos"]:
                if vid in fout["videos"]:
                    del fout["videos"][vid]
                fin["videos"].copy(vid, fout["videos"])
    fout.close()

def _write_dataset_map(out_h5):
    """Crea /meta/dataset_map con pares (video_id, dataset_id) para splits rápidos."""
    with h5py.File(out_h5, "a") as f:
        if "videos" not in f:
            return
        vids, dsets = [], []
        for vid, g in f["videos"].items():
            vids.append(vid)
            dsets.append(str(g.attrs.get("dataset_id", "")))
        dt = h5py.string_dtype(encoding="utf-8")
        rec = np.rec.fromarrays([np.array(vids, dtype=dt), np.array(dsets, dtype=dt)],
                                names=("video_id","dataset_id"))
        meta = f["meta"]
        if "dataset_map" in meta:
            del meta["dataset_map"]
        meta.create_dataset("dataset_map", data=rec, compression="gzip")

def main():
    ap = argparse.ArgumentParser(description="Multiproceso por video con merge final a un único H5")
    default_manifest = str(ROOT / "1.LabelAlignment" / "Manifests" / "videos.csv")
    default_out = str(ROOT / "prepared" / "h5" / "frames_mediapipe_all.h5")
    ap.add_argument("--manifest", default=default_manifest)
    ap.add_argument("--out-h5", default=default_out)
    ap.add_argument("--estimator", default="mediapipe")
    ap.add_argument("--coord-space", default="image", choices=["image","world"])
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-proc", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-parts", action="store_true", help="No borrar los shards luego del merge")
    args = ap.parse_args()

    # Cargar filas del manifest
    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break

    if not rows:
        raise SystemExit("No hay filas en el manifest")

    nprocs = min(args.max_proc, cpu_count())
    print(f"Start method: {get_start_method()}  CPU: {cpu_count()}  Workers: {nprocs}")
    print(f"Videos totales: {len(rows)}")

    results = []
    parts_ok = []
    per_ds = {}

    # Procesamiento paralelo por video con barra de progreso global
    with Pool(processes=nprocs,
              initializer=_init_worker,
              initargs=(args.coord_space, args.estimator, args.stride)) as pool:
        for res in tqdm(pool.imap_unordered(_process_one_video, rows),
                        total=len(rows), desc="Videos", unit="vid"):
            results.append(res)
            if res["ok"]:
                parts_ok.append(res["part"])
                per_ds[res["dataset_id"]] = per_ds.get(res["dataset_id"], 0) + 1

    # Resumen por dataset
    print("\nResumen por dataset:")
    for ds, cnt in sorted(per_ds.items()):
        print(f" - {ds}: {cnt} videos ok")

    # Merge a un único H5
    print(f"\nFusionando {len(parts_ok)} shards en {args.out_h5}")
    _merge_parts(parts_ok, args.out_h5, args.estimator, args.coord_space, show_bar=True)
    _write_dataset_map(args.out_h5)
    print("Listo ->", args.out_h5)

    # Limpieza opcional
    if not args.keep-parts:
        try:
            shutil.rmtree(PARTS_DIR)
            print(f"Eliminados shards en {PARTS_DIR}")
        except Exception as e:
            print(f"[WARN] No se pudo borrar {PARTS_DIR}: {e}")

if __name__ == "__main__":
    # from multiprocessing import freeze_support; freeze_support()  # si empaquetas
    main()
