"""
============================================================
 STEP 2 — DATA AUGMENTATION  (v6, 24 dims)
 Temporal Vision-Language Model | Semaphore Translation
============================================================
 Input  : poses_raw.csv
 Output : poses_augmented.csv

 Changes from v5:
   - 24-dim feature vector. Removed the redundant lateral
     wrist features (mathematically identical to the
     normalised wrist X-coords already in the coord block).
   - aug_joint_noise now uses L2 shoulder width (matching
     extract_poses.py and compare_models.py) instead of the
     1D abs(ls.x - rs.x), which under-scaled the noise after
     rotation augmentation.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV         = Path("poses_raw.csv")
OUTPUT_CSV        = Path("poses_augmented.csv")

N_AUGMENTS        = 8
ROTATION_RANGE    = 15.0
TRANSLATION_MAX   = 0.05
DEPTH_SCALE_RANGE = 0.10
JOINT_NOISE_XY    = 0.020    # 2% of shoulder width
JOINT_NOISE_Z     = 0.010    # 1% of shoulder width

LANDMARK_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
]
COORD_COLS = [f"{n}_{ax}" for n in LANDMARK_ORDER for ax in ("x", "y", "z")]
ANGLE_COLS = [
    "left_upper_arm_angle", "right_upper_arm_angle",
    "left_forearm_angle",   "right_forearm_angle",
    "left_elbow_bend",      "right_elbow_bend",
]
FEATURE_COLS = COORD_COLS + ANGLE_COLS      # 24 dims
LABEL_COL    = "label"
SOURCE_COL   = "source_id"

# ─── Angle Recomputation ─────────────────────────────────────────────────────

def _angle_vs_vertical(p1, p2) -> float:
    return float(np.degrees(np.arctan2(p2[0]-p1[0], p2[1]-p1[1])))

def _elbow_bend(s, e, w) -> float:
    v1 = s - e; v2 = w - e
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def recompute_derived(c: dict) -> list[float]:
    return [
        _angle_vs_vertical(c["left_shoulder"][:2],  c["left_elbow"][:2]),
        _angle_vs_vertical(c["right_shoulder"][:2], c["right_elbow"][:2]),
        _angle_vs_vertical(c["left_elbow"][:2],     c["left_wrist"][:2]),
        _angle_vs_vertical(c["right_elbow"][:2],    c["right_wrist"][:2]),
        _elbow_bend(c["left_shoulder"][:2],  c["left_elbow"][:2],  c["left_wrist"][:2]),
        _elbow_bend(c["right_shoulder"][:2], c["right_elbow"][:2], c["right_wrist"][:2]),
    ]

# ─── Row ↔ Coord Dict ─────────────────────────────────────────────────────────

def row_to_coords(row: pd.Series) -> dict:
    return {
        name: np.array([row[f"{name}_x"], row[f"{name}_y"], row[f"{name}_z"]],
                       dtype=np.float32)
        for name in LANDMARK_ORDER
    }

def coords_to_record(coords: dict, label: int, source_id: int) -> dict:
    rec = {}
    for name, xyz in coords.items():
        rec[f"{name}_x"] = float(xyz[0])
        rec[f"{name}_y"] = float(xyz[1])
        rec[f"{name}_z"] = float(xyz[2])
    for col, val in zip(ANGLE_COLS, recompute_derived(coords)):
        rec[col] = val
    rec[LABEL_COL]  = label
    rec[SOURCE_COL] = source_id
    return rec

# ─── Augmentation Primitives ─────────────────────────────────────────────────

def aug_rotation(coords, rng):
    theta  = np.deg2rad(rng.uniform(-ROTATION_RANGE, ROTATION_RANGE))
    c, s   = np.cos(theta), np.sin(theta)
    R      = np.array([[c, -s], [s, c]], dtype=np.float32)
    ls, rs = coords["left_shoulder"], coords["right_shoulder"]
    pivot  = np.array([(ls[0]+rs[0])/2, (ls[1]+rs[1])/2], dtype=np.float32)
    return {
        name: np.array([*(R @ (xyz[:2] - pivot) + pivot), xyz[2]], dtype=np.float32)
        for name, xyz in coords.items()
    }

def aug_translation(coords, rng):
    dx, dy = rng.uniform(-TRANSLATION_MAX, TRANSLATION_MAX, size=2)
    return {
        name: np.array([xyz[0]+dx, xyz[1]+dy, xyz[2]], dtype=np.float32)
        for name, xyz in coords.items()
    }

def aug_depth(coords, rng):
    scale = rng.uniform(1.0 - DEPTH_SCALE_RANGE, 1.0 + DEPTH_SCALE_RANGE)
    return {
        name: np.array([xyz[0], xyz[1], xyz[2]*scale], dtype=np.float32)
        for name, xyz in coords.items()
    }

def aug_joint_noise(coords, rng):
    """
    Independently perturb each landmark by a small Gaussian amount,
    scaled to shoulder width so it's distance-invariant.

    sw uses the 2D L2 norm (matching extract_poses.py and the
    normalisation in _normalise). Using abs(ls.x - rs.x) here under-
    estimates sw whenever the pose has been rotated, which silently
    shrinks the noise magnitude on rotated copies.
    """
    ls = coords["left_shoulder"]
    rs = coords["right_shoulder"]
    sw = float(np.linalg.norm(ls[:2] - rs[:2])) + 1e-8
    sigma_xy = JOINT_NOISE_XY * sw
    sigma_z  = JOINT_NOISE_Z  * sw
    return {
        name: np.array([
            xyz[0] + rng.normal(0, sigma_xy),
            xyz[1] + rng.normal(0, sigma_xy),
            xyz[2] + rng.normal(0, sigma_z),
        ], dtype=np.float32)
        for name, xyz in coords.items()
    }

_AUGMENTERS = [aug_rotation, aug_translation, aug_depth, aug_joint_noise]

# ─── Augmentation Loop ────────────────────────────────────────────────────────

def augment_sample(row: pd.Series, source_id: int,
                   rng: np.random.Generator) -> dict:
    coords = row_to_coords(row)
    for fn in _AUGMENTERS:
        if rng.random() < 0.5:
            coords = fn(coords, rng)
    return coords_to_record(coords, int(row[LABEL_COL]), source_id)

def augment_dataset(df_raw: pd.DataFrame,
                    n_augments: int = N_AUGMENTS,
                    seed: int = 42) -> pd.DataFrame:
    rng     = np.random.default_rng(seed)
    records = []
    total   = len(df_raw)

    # Originals — each gets its row index as source_id
    for i, (_, row) in enumerate(df_raw.iterrows()):
        rec = row.to_dict()
        rec[SOURCE_COL] = i
        records.append(rec)

    # Augmented copies — share the original's source_id
    for i, (_, row) in enumerate(df_raw.iterrows()):
        if (i+1) % 100 == 0 or (i+1) == total:
            print(f"  Augmenting {i+1}/{total} …", end="\r")
        for _ in range(n_augments):
            records.append(augment_sample(row, i, rng))
    print()

    cols   = FEATURE_COLS + [LABEL_COL, SOURCE_COL]
    df_aug = pd.DataFrame(records, columns=cols)
    return df_aug.sample(frac=1, random_state=seed).reset_index(drop=True)

# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  STEP 2 — Coordinate-Space Augmentation  (v6, 24 dims)")
    print("=" * 60)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"'{INPUT_CSV}' not found — run extract_poses.py first.")

    df_raw = pd.read_csv(INPUT_CSV)
    print(f"[INFO] Raw samples   : {len(df_raw)}")
    print(f"[INFO] Feature dims  : {len(FEATURE_COLS)}")
    print(f"[INFO] Augment ×{N_AUGMENTS}   : ~{len(df_raw)*(N_AUGMENTS+1)} total")
    print(f"[INFO] Augmenters    : "
          f"{', '.join(fn.__name__ for fn in _AUGMENTERS)}")

    df_aug = augment_dataset(df_raw)
    print(f"[INFO] Final count   : {len(df_aug)}")
    print(f"[INFO] source_id rng : {df_aug[SOURCE_COL].min()}–{df_aug[SOURCE_COL].max()}  "
          f"({df_aug[SOURCE_COL].nunique()} unique sources)")

    df_aug.to_csv(OUTPUT_CSV, index=False)
    print(f"[✓] Saved → '{OUTPUT_CSV}'")

if __name__ == "__main__":
    main()