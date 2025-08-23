#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Paso 1: LabelAlignment (ConsensusBuilder)

Lee anotaciones múltiples por video (PSKUS) y produce un rastro único por frame:
- is_washing_cons ∈ {0,1}  (−1 = indeterminado por empate)
- movement_cons   ∈ {0..7} (−1 = indeterminado por empate o no_washing)
- transition_mask ∈ {0,1}  (1 en ±transition_frames alrededor de cambios)

Layout esperado en disco:
  PSKUS_dataset/
    DataSetX/
      Videos/<video>.mp4
      Annotations/Annotator1/<video>.json
      Annotations/Annotator2/<video>.json
      ...

Salida:
  LAIA-Net/1.LabelAlignment/Consensus/DataSetX/<video>.csv
  LAIA-Net/1.LabelAlignment/Manifests/videos.csv

python3 builder.py --datasets-root ../../PSKUS_dataset --out-root .
"""

import os
import csv
import json
import argparse
import cv2
import pandas as pd
import numpy as np

def parse_args():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Por defecto, PSKUS_dataset está al mismo nivel que LAIA-Net
    default_datasets_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir, "PSKUS_dataset"))
    parser = argparse.ArgumentParser(description="ConsensusBuilder for PSKUS")
    parser.add_argument("--datasets-root", type=str, default=default_datasets_root,
                        help="Ruta a PSKUS_dataset")
    parser.add_argument("--out-root", type=str, default=script_dir,
                        help="Carpeta base de salida (por defecto esta misma)")
    parser.add_argument("--fps", type=int, default=30, help="FPS nominal")
    parser.add_argument("--transition-frames", type=int, default=15,
                        help="Frames a enmascarar antes y despues de cada cambio de movimiento")
    parser.add_argument("--map7to0", action="store_true",
                        help="Mapea clase 7 a 0 durante el voto")
    parser.add_argument("--min-annotators", type=int, default=1,
                        help="Minimo de anotadores requeridos")
    return parser.parse_args()

def list_datasets(datasets_root):
    out = []
    if not os.path.isdir(datasets_root):
        return out
    for name in sorted(os.listdir(datasets_root)):
        p = os.path.join(datasets_root, name)
        if os.path.isdir(p) and name.lower().startswith("dataset"):
            out.append(p)
    return out

def list_videos(dataset_dir):
    vids = []
    vdir = os.path.join(dataset_dir, "Videos")
    if not os.path.isdir(vdir):
        return vids
    for name in sorted(os.listdir(vdir)):
        if name.lower().endswith(".mp4"):
            vids.append(os.path.join(vdir, name))
    return vids

def find_annotation_files(dataset_dir, video_stem):
    ann_dir = os.path.join(dataset_dir, "Annotations")
    if not os.path.isdir(ann_dir):
        return []
    out = []
    for name in sorted(os.listdir(ann_dir)):
        p = os.path.join(ann_dir, name)
        if os.path.isdir(p) and name.lower().startswith("annotator"):
            jf = os.path.join(p, video_stem + ".json")
            cf = os.path.join(p, video_stem + ".csv")
            if os.path.isfile(jf):
                out.append(jf)
            elif os.path.isfile(cf):
                out.append(cf)
    return out

def probe_video_meta(video_path):
    """
    Given a video path, returns a dictionary with the number of frames, width, height, and fps.
    If the video cannot be opened, returns an empty dictionary.
    If the fps is not an integer, rounds it to the nearest integer.
    Args:
        video_path (str): The path to the video file.
    Returns:
        dict: A dictionary with the number of frames, width, height, and fps.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"num_frames": "", "w": "", "h": "", "fps": ""}
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    try:
        if abs(fps - round(fps)) < 0.01:
            fps = int(round(fps))
    except:
        pass
    return {"num_frames": num_frames, "w": w, "h": h, "fps": fps}

def read_annotations_file(path):
    try:
        if path.lower().endswith(".json"):
            with open(path,"r") as f:
                data = json.load(f)
            if isinstance(data, dict) and "labels" in data:
                labels = data["labels"]
            elif isinstance(data, list):
                labels = data
            else:
                return None
            out = []
            for it in labels:
                iw = int(it.get("is_washing", 0))
                cd = int(it.get("code", -1))
                out.append({"is_washing": iw, "code": cd})
            return out
        else:
            out = []
            df = pd.read_csv(path)
            for i, row in df.iterrows():
                iw = int(row.get("is_washing", 0))
                cd = int(row.get("code", -1))
                out.append({"is_washing": iw, "code": cd})
            return out
    except Exception as e:
        print("[WARN] No pude leer {}: {}".format(os.path.basename(path), e))
        return None

def load_all_annotators(paths):
    tracks = []
    for p in paths:
        seq = read_annotations_file(p)
        # print("[INFO] {} anotaciones leidas de {}".format(len(seq),p))
        if seq and len(seq) > 0:
            tracks.append(seq)
    if len(tracks) == 0:
        return []
    min_len = min(len(t) for t in tracks)
    tracks = [t[:min_len] for t in tracks]
    return tracks

def majority_vote_int(values, valid_set=None):
    counts = {}
    total = 0
    for v in values:
        if valid_set is not None and v not in valid_set:
            continue
        counts[v] = counts.get(v, 0) + 1
        total += 1
    if total == 0:
        return -1
    best_class = None
    best_count = -1
    for cls, cnt in counts.items():
        if cnt > best_count:
            best_class, best_count = cls, cnt
    n_needed = (total // 2) + 1
    if best_count < n_needed:
        return -1
    return int(best_class)
def apply_transition_mask(labels, radius):
    n = len(labels)
    mask = [0] * n
    clean = []
    for i in range(n):
        if labels[i] != -1:
            clean.append((i, labels[i]))
    if len(clean) == 0:
        return mask
    prev_i, prev_y = clean[0]
    for i, y in clean[1:]:
        if y != prev_y:
            a = max(0, i - radius)
            b = min(n, i + radius + 1)
            for j in range(a, b):
                mask[j] = 1
        prev_i, prev_y = i, y
    return mask

def consensus_for_video(tracks, map7to0):
    # Devuelve is_washing_cons, movement_cons, votes_hist
    if len(tracks) == 0:
        return [], [], []
    T = min(len(t) for t in tracks)
    isw_cons = [-1] * T
    mov_cons = [-1] * T
    votes_hist = [[0]*8 for _ in range(T)]
    for f in range(T):
        iw_votes = [int(t[f]["is_washing"]) for t in tracks]
        iw = majority_vote_int(iw_votes, valid_set=set([0,1]))
        isw_cons[f] = iw
        # votos de movimiento solo donde is_washing=1
        mvotes = []
        for t in tracks:
            if int(t[f]["is_washing"]) == 1:
                cd = int(t[f]["code"])
                if map7to0 and cd == 7:
                    cd = 0
                if 0 <= cd <= 7:
                    mvotes.append(cd)
        for cd in mvotes:
            votes_hist[f][cd] += 1
        if iw == 1:
            mv = majority_vote_int(mvotes, valid_set=set(range(8)))
        else:
            mv = -1
        mov_cons[f] = mv
    return isw_cons, mov_cons, votes_hist

def ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def write_consensus_csv(out_csv, isw, mov, mask, votes_hist, n_annotators):
    ensure_dir(os.path.dirname(out_csv))
    header = ["frame_idx","is_washing_cons","movement_cons","transition_mask","n_annotators"] + \
             ["votes_{}".format(i) for i in range(8)]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(isw)):
            row = [i, isw[i], mov[i], mask[i], n_annotators] + votes_hist[i]
            w.writerow(row)


def append_manifest_row(manifest_csv, row, wrote_header):
    ensure_dir(os.path.dirname(manifest_csv))
    header = ["dataset_id","video_id","camera_id","fps","width","height","num_frames","path_video","path_consensus"]
    file_exists = os.path.isfile(manifest_csv)
    with open(manifest_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not file_exists and not wrote_header:
            w.writeheader()
        w.writerow(row)
def main():
    args = parse_args()
    datasets_root = args.datasets_root
    out_root = args.out_root
    out_consensus_root = os.path.join(out_root, "Consensus")
    manifest_csv = os.path.join(out_root, "Manifests", "videos.csv")

    datasets = list_datasets(datasets_root)
    if len(datasets) == 0:
        print("[ERROR] No se encontraron DataSet* en {}".format(datasets_root))
        return
    wrote_header = False

    for dset in datasets:
        videos = list_videos(dset)
        print("[INFO] Procesando {} videos de {}".format(len(videos), dset))
        for vpath in videos:
            video_stem = os.path.splitext(os.path.basename(vpath))[0]
            ann_files = find_annotation_files(dset, video_stem)
            if len(ann_files) < args.min_annotators:
                print("[INFO] {}: {} anotador(es). Omitiendo (min={}).".format(video_stem, len(ann_files), args.min_annotators))
                continue
            tracks = load_all_annotators(ann_files)
            if len(tracks) < args.min_annotators:
                print("[INFO] {}: anotaciones insuficientes. Omitiendo.".format(video_stem))
                continue
            isw_cons, mov_cons, votes_hist = consensus_for_video(tracks, map7to0=args.map7to0)
            trans_mask = apply_transition_mask(mov_cons, radius=args.transition_frames)
           
            out_csv = os.path.join(out_consensus_root, os.path.basename(dset), video_stem + ".csv")
            write_consensus_csv(out_csv, isw_cons, mov_cons, trans_mask, votes_hist, n_annotators=len(tracks))


            meta = probe_video_meta(vpath)
            camera_id = ""
            parts = video_stem.split("_")

            for token in parts:
                if token.lower().startswith("camera"):
                    camera_id = token
                    break
            manifest_row = {
                "dataset_id": os.path.basename(dset),
                "video_id": video_stem,
                "camera_id": camera_id,
                "fps": int(meta["fps"]) if meta["fps"] != "" and meta["fps"] is not None else args.fps,
                "width": meta["w"],
                "height": meta["h"],
                "num_frames": meta["num_frames"],
                "path_video": vpath,
                "path_consensus": out_csv
            }
            append_manifest_row(manifest_csv, manifest_row, wrote_header)
            wrote_header = True
            
            print("[OK] {}/{} -> {}".format(os.path.basename(dset), video_stem, out_csv))

    print("[DONE] Label Alignment completed")


if __name__ == "__main__":
    main()
