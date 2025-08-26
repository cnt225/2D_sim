#!/usr/bin/env python3
import os, sys, json, glob
from pathlib import Path
import argparse
import torch

# Ensure local utils import
sys.path.append(str(Path(__file__).resolve().parents[1]))
from utils.motion_utils import twist_to_se3_matrix


def load_json(path: str):
    with open(path, 'r') as f:
        return json.load(f)


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f)


def convert_one(in_path: str, out_dir: str, overwrite: bool = False) -> str:
    data = load_json(in_path)
    path = data.get('path', {})
    twists = path.get('data', [])  # list of [6]
    timestamps = path.get('timestamps', [])
    dt_val = data.get('dt', None)

    if not twists:
        return ''

    # Determine dt
    if dt_val is None:
        if isinstance(timestamps, list) and len(timestamps) >= 2:
            dt_val = float(timestamps[1] - timestamps[0])
        else:
            dt_val = 0.01

    twists_t = torch.tensor(twists, dtype=torch.float32)  # [N,6]
    # Keep original twist velocities (rad/s, m/s) for direct velocity learning
    # No dt scaling - model will learn velocities directly
    delta_T = twist_to_se3_matrix(twists_t).cpu().numpy().tolist()  # [N,4,4]

    # Write out JSON: preserve metadata; add path['delta_T']
    out_data = data
    if 'path' not in out_data:
        out_data['path'] = {}
    out_data['path']['delta_T'] = delta_T
    out_rel = os.path.basename(in_path)
    out_path = os.path.join(out_dir, out_rel)

    if not overwrite and os.path.exists(out_path):
        return out_path

    save_json(out_path, out_data)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, '*.json')))
    if args.limit > 0:
        files = files[:args.limit]

    print(f'Found {len(files)} files')
    os.makedirs(args.out_dir, exist_ok=True)
    ok = 0
    for i, fp in enumerate(files, 1):
        try:
            outp = convert_one(fp, args.out_dir, overwrite=args.overwrite)
            ok += 1
            if i % 50 == 0:
                print(f'Converted {i}/{len(files)}')
        except Exception as e:
            print(f'Failed {fp}: {e}')
    print(f'Done. Converted: {ok}/{len(files)}')

if __name__ == '__main__':
    main()
