"""
============================================================
 MODEL COMPARISON — DNN vs Random Forest vs LSTM  (v2, 24 dims)
 Temporal Vision-Language Model | Semaphore Translation
============================================================
 Research questions:
   Q1.  Is a DNN justified on 24-dim features, or does Random
        Forest match it at lower cost?
   Q2.  Does a native temporal model (LSTM) outperform a static
        DNN + temporal smoothing for this task?

 Inputs:
   poses_raw.csv         (for LSTM — temporal windows)
   poses_augmented.csv   (for DNN, RF — per-frame, includes source_id)

 Outputs (./model_comparison/):
   report.md                — readable summary (open this first)
   metrics.json             — raw numbers, all runs
   metrics_summary.csv      — per-model summary (mean ± std)
   per_class_f1.csv         — per-class F1 for each model
   confusion_<model>.png    — confusion matrix per model
   accuracy_comparison.png  — bar chart with error bars
   latency_comparison.png   — bar chart, log scale
   training_curves.png      — DNN vs LSTM training loss

 Methodology:
   • Within a run, all 3 models train on the SAME source-frame
     train/val/test partition (group-aware, group=source_id —
     identical to train_model.py's strategy).
   • LSTM windows are assigned by the source_id of the window's
     center frame. Augmented copies of a window share that group,
     so no augmentation leak across splits. (See report.md for a
     caveat on frame-level overlap between adjacent windows.)
   • N_RUNS=3 different partition seeds → mean ± std for variance.
   • Same StandardScaler fit on training data only.
   • Inference latency = wall-clock for predicting a single sample,
     averaged over 1000 calls after a 100-call warmup. This is the
     metric that matters for live use (~30 fps target).

 Changes from v1:
   - 24-dim features (lateral wrist features dropped — they were
     duplicates of normalised wrist X-coords).
   - DNN and LSTM training now use class_weight='balanced',
     matching train_model.py. Previously only RF used it, which
     unfairly disadvantaged the neural-net models against RF.
"""

import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing      import StandardScaler, LabelEncoder
from sklearn.model_selection    import GroupShuffleSplit
from sklearn.ensemble           import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics            import (
    accuracy_score, f1_score,
    confusion_matrix, precision_recall_fscore_support,
)

import tensorflow as tf
from tensorflow                 import keras
from tensorflow.keras           import layers, callbacks

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_CSV  = Path("poses_raw.csv")
AUG_CSV  = Path("poses_augmented.csv")
OUT_DIR  = Path("model_comparison")

N_RUNS         = 3            # repetitions for variance estimation (set 1 for quick check)
N_CLASSES      = 27

# LSTM windowing
WINDOW_LEN     = 10           # frames per window  (= 1.0 s @ 10 fps sampling)
WINDOW_STRIDE  = 3            # frames between windows
LSTM_AUGS      = 4            # augmentations per window (≈ matches DNN/RF aug factor)

# Architectures
DNN_HIDDEN     = [128, 64, 32]
DROPOUT        = 0.20
LSTM_UNITS     = 64
RF_TREES       = 300

# Training
LR             = 1e-3
DNN_EPOCHS     = 80
DNN_BATCH      = 64
DNN_PATIENCE   = 12
LSTM_EPOCHS    = 60
LSTM_BATCH     = 64
LSTM_PATIENCE  = 10

# LSTM augmentation magnitudes (mirror augment_data.py constants)
ROT_RANGE_DEG  = 15.0
TRANS_MAX      = 0.05
DEPTH_SCALE    = 0.10
JOINT_NOISE_XY = 0.020        # per-frame, independently drawn
JOINT_NOISE_Z  = 0.010

# Latency
LATENCY_WARMUP = 100
LATENCY_TRIALS = 1000

LABEL_COL  = "label"
SOURCE_COL = "source_id"

LANDMARK_ORDER = [
    "left_shoulder", "right_shoulder",
    "left_elbow",    "right_elbow",
    "left_wrist",    "right_wrist",
]
COORD_COLS = [f"{n}_{a}" for n in LANDMARK_ORDER for a in ("x", "y", "z")]
ANGLE_COLS = [
    "left_upper_arm_angle", "right_upper_arm_angle",
    "left_forearm_angle",   "right_forearm_angle",
    "left_elbow_bend",      "right_elbow_bend",
]
FEATURE_COLS = COORD_COLS + ANGLE_COLS                # 24 dims

DISPLAY_NAME = {**{i+1: chr(ord("A")+i) for i in range(26)}, 27: "SPC"}

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def derive_video_ids(df_raw: pd.DataFrame) -> np.ndarray:
    """Infer a video_id per row from consecutive runs of the same label.

    poses_raw.csv is produced in video order; a change in label between
    consecutive rows marks a new video. Used only for windowing (sequences
    must come from one video) — not for splitting.
    """
    labels = df_raw[LABEL_COL].values
    vid_ids = np.zeros(len(labels), dtype=np.int32)
    cur = 0
    for i in range(1, len(labels)):
        if labels[i] != labels[i-1]:
            cur += 1
        vid_ids[i] = cur
    return vid_ids


def load_per_frame() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_raw, source_id) for DNN/RF training.

    source_id is the group used for splitting (one group per original raw
    frame; all augmented copies share their source's group). This is the
    same grouping train_model.py uses, ensuring DNN/RF here are evaluated
    under conditions identical to your current pipeline.
    """
    if not AUG_CSV.exists():
        raise FileNotFoundError(f"'{AUG_CSV}' not found — run augment_data.py first.")
    df_aug = pd.read_csv(AUG_CSV)
    if SOURCE_COL not in df_aug.columns:
        raise ValueError(f"'{AUG_CSV}' must contain a '{SOURCE_COL}' column. Re-run augment_data.py.")

    X = df_aug[FEATURE_COLS].values.astype(np.float32)
    y = df_aug[LABEL_COL].values.astype(int)
    s = df_aug[SOURCE_COL].values.astype(int)
    return X, y, s


# ─── LSTM windowing + per-window augmentation ────────────────────────────────

def _rotate_xy(xy: np.ndarray, theta_rad: float, pivot: np.ndarray) -> np.ndarray:
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return (xy - pivot) @ R.T + pivot


def _apply_window_aug(window_coords: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply augmentation to a (T, 6, 3) window.

    Gesture-level transforms (rotation, translation, depth scale) are
    drawn ONCE and applied uniformly across the whole window — the gesture
    itself doesn't change angle mid-window.

    Per-joint XY/Z noise is drawn INDEPENDENTLY per frame — this is exactly
    the kind of frame-to-frame jitter an LSTM should learn to smooth over.
    """
    w = window_coords.copy()
    T = w.shape[0]

    # 2D L2 shoulder width, averaged across the window (matches
    # extract_poses.py's normalisation)
    ls_xy = w[:, 0, :2]
    rs_xy = w[:, 1, :2]
    sw = float(np.mean(np.linalg.norm(ls_xy - rs_xy, axis=1))) + 1e-8

    # gesture-level transforms (consistent across frames)
    if rng.random() < 0.5:
        theta = np.deg2rad(rng.uniform(-ROT_RANGE_DEG, ROT_RANGE_DEG))
        pivot = (w[:, 0, :2].mean(axis=0) + w[:, 1, :2].mean(axis=0)) / 2.0
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]], dtype=np.float32)
        # (T, 6, 2) − pivot, rotate, + pivot — broadcasts cleanly
        w[:, :, :2] = (w[:, :, :2] - pivot) @ R.T + pivot
    if rng.random() < 0.5:
        dx, dy = rng.uniform(-TRANS_MAX, TRANS_MAX, size=2)
        w[:, :, 0] += dx
        w[:, :, 1] += dy
    if rng.random() < 0.5:
        scale = rng.uniform(1.0 - DEPTH_SCALE, 1.0 + DEPTH_SCALE)
        w[:, :, 2] *= scale

    # per-frame independent joint noise
    if rng.random() < 0.5:
        sigma_xy = JOINT_NOISE_XY * sw
        sigma_z  = JOINT_NOISE_Z  * sw
        w[:, :, :2] += rng.normal(0, sigma_xy, size=(T, 6, 2)).astype(np.float32)
        w[:, :,  2] += rng.normal(0, sigma_z,  size=(T, 6)).astype(np.float32)

    return w


def _coords_window_to_features(window_coords: np.ndarray) -> np.ndarray:
    """Convert a (T, 6, 3) window to a (T, 24) feature window.

    Replicates extract_poses.py's feature computation per frame:
      18 normalised XYZ + 6 joint angles.

    Fully vectorised over T. Mathematically identical to the per-frame
    extract_poses.py logic; just avoids a Python for-loop over time steps
    so that load_temporal() runs in ~1 s instead of ~60 s on real data.
    """
    # split joints: each is (T, 3)
    ls  = window_coords[:, 0]
    rs  = window_coords[:, 1]
    le  = window_coords[:, 2]
    re_ = window_coords[:, 3]
    lw  = window_coords[:, 4]
    rw  = window_coords[:, 5]

    mid  = (ls + rs) * 0.5                                            # (T, 3)
    sw   = np.linalg.norm(ls[:, :2] - rs[:, :2], axis=1) + 1e-8       # (T,)
    sw_e = sw[:, None]                                                # (T, 1)

    T = window_coords.shape[0]
    out = np.empty((T, 24), dtype=np.float32)
    out[:,  0:3]  = (ls  - mid) / sw_e
    out[:,  3:6]  = (rs  - mid) / sw_e
    out[:,  6:9]  = (le  - mid) / sw_e
    out[:,  9:12] = (re_ - mid) / sw_e
    out[:, 12:15] = (lw  - mid) / sw_e
    out[:, 15:18] = (rw  - mid) / sw_e

    def ang(a, b):                                          # atan2(dx, dy)
        return np.degrees(np.arctan2(b[:, 0] - a[:, 0], b[:, 1] - a[:, 1]))

    out[:, 18] = ang(ls,  le)
    out[:, 19] = ang(rs,  re_)
    out[:, 20] = ang(le,  lw)
    out[:, 21] = ang(re_, rw)

    def bend(a, b, c):                                      # XY only
        v1 = a[:, :2] - b[:, :2]
        v2 = c[:, :2] - b[:, :2]
        dot = (v1 * v2).sum(axis=1)
        n1  = np.linalg.norm(v1, axis=1) + 1e-8
        n2  = np.linalg.norm(v2, axis=1) + 1e-8
        return np.degrees(np.arccos(np.clip(dot / (n1 * n2), -1.0, 1.0)))

    out[:, 22] = bend(ls, le,  lw)
    out[:, 23] = bend(rs, re_, rw)

    return out


def load_temporal() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LSTM training data: (N_windows, T, 24), labels, group_id.

    Uses poses_raw.csv directly (frames in temporal order within each video).
    Slides a window of length WINDOW_LEN with stride WINDOW_STRIDE through
    each video. Each window is augmented LSTM_AUGS times (the original is
    included too).

    group_id = source_id of the window's CENTER frame. This is the same
    group concept DNN/RF use (one group per raw frame), so matched_split
    can put the same source frames on the same side of the partition for
    all three models.
    """
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"'{RAW_CSV}' not found — run extract_poses.py first.")
    print(f"       reading {RAW_CSV}…", flush=True)
    df_raw = pd.read_csv(RAW_CSV)
    print(f"       {len(df_raw)} rows read", flush=True)

    coord_arr = df_raw[COORD_COLS].values.astype(np.float32).reshape(-1, 6, 3)
    labels    = df_raw[LABEL_COL].values.astype(int)
    vid_ids   = derive_video_ids(df_raw)
    raw_idx   = np.arange(len(df_raw), dtype=np.int32)             # acts as source_id

    unique_vids = np.unique(vid_ids)
    n_vids = len(unique_vids)
    print(f"       derived {n_vids} video segments (expecting ~27); "
          f"building windows of length {WINDOW_LEN}…", flush=True)
    if n_vids > 200:
        print(f"       [WARN] {n_vids} segments is far more than expected — "
              "is poses_raw.csv in original video order? Frames must be "
              "consecutive per label for windowing to make sense.", flush=True)

    X_seq, y_seq, g_seq = [], [], []
    rng = np.random.default_rng(0)

    for i, vid in enumerate(unique_vids, 1):
        mask = vid_ids == vid
        coords = coord_arr[mask]                                    # (n_frames, 6, 3)
        sources = raw_idx[mask]
        label = int(labels[mask][0])
        n = coords.shape[0]
        if n < WINDOW_LEN:
            continue
        for start in range(0, n - WINDOW_LEN + 1, WINDOW_STRIDE):
            w_coords = coords[start:start + WINDOW_LEN]             # (T, 6, 3)
            center_source = int(sources[start + WINDOW_LEN // 2])
            # original
            X_seq.append(_coords_window_to_features(w_coords))
            y_seq.append(label)
            g_seq.append(center_source)
            # augmentations — same group as original (no leak)
            for _ in range(LSTM_AUGS):
                w_aug = _apply_window_aug(w_coords, rng)
                X_seq.append(_coords_window_to_features(w_aug))
                y_seq.append(label)
                g_seq.append(center_source)
        if i % 5 == 0 or i == n_vids:
            print(f"       …video {i}/{n_vids}  frames={n}  "
                  f"windows_so_far={len(X_seq)}", flush=True)

    if not X_seq:
        raise RuntimeError(
            "No LSTM windows produced. Either every video has < "
            f"{WINDOW_LEN} frames, or poses_raw.csv isn't in temporal order."
        )

    print(f"       stacking {len(X_seq)} windows into final array…", flush=True)
    X_seq = np.stack(X_seq, axis=0)
    y_seq = np.array(y_seq, dtype=int)
    g_seq = np.array(g_seq, dtype=np.int32)
    print(f"       done. X_seq shape: {X_seq.shape}", flush=True)
    return X_seq, y_seq, g_seq


# ═════════════════════════════════════════════════════════════════════════════
# SPLITTING
# ═════════════════════════════════════════════════════════════════════════════

def matched_split(all_groups: np.ndarray, seed: int) -> tuple[set[int], set[int], set[int]]:
    """Partition unique group ids (source_ids) into train/val/test sets.

    Returns sets of group ids. Samples are then assigned to splits by group
    membership, identical for DNN/RF (whose groups are explicit source_ids)
    and LSTM (whose group is the window's center source_id).
    """
    unique_g = np.unique(all_groups)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    tv_idx, te_idx = next(gss1.split(unique_g, groups=unique_g))
    test_g = set(int(g) for g in unique_g[te_idx])
    rest_g = unique_g[tv_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20/0.90, random_state=seed+1)
    tr_idx, va_idx = next(gss2.split(rest_g, groups=rest_g))
    val_g   = set(int(g) for g in rest_g[va_idx])
    train_g = set(int(g) for g in rest_g[tr_idx])
    return train_g, val_g, test_g


def apply_split(groups: np.ndarray, train_g, val_g, test_g):
    tr = np.isin(groups, list(train_g))
    va = np.isin(groups, list(val_g))
    te = np.isin(groups, list(test_g))
    return tr, va, te


def balanced_class_weight(y_train: np.ndarray) -> dict[int, float]:
    """Compute sklearn-style balanced class weights for the training labels."""
    classes = np.unique(y_train)
    w = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(c): float(wi) for c, wi in zip(classes, w)}


# ═════════════════════════════════════════════════════════════════════════════
# MODELS
# ═════════════════════════════════════════════════════════════════════════════

def build_dnn(input_dim: int, n_classes: int) -> keras.Model:
    m = keras.Sequential(name="DNN")
    m.add(layers.Input(shape=(input_dim,)))
    for i, u in enumerate(DNN_HIDDEN, 1):
        m.add(layers.Dense(u))
        m.add(layers.BatchNormalization())
        m.add(layers.Activation("relu"))
        m.add(layers.Dropout(DROPOUT))
    m.add(layers.Dense(n_classes, activation="softmax"))
    m.compile(optimizer=keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_lstm(time_steps: int, input_dim: int, n_classes: int) -> keras.Model:
    m = keras.Sequential(name="LSTM")
    m.add(layers.Input(shape=(time_steps, input_dim)))
    m.add(layers.LSTM(LSTM_UNITS, return_sequences=False))
    m.add(layers.BatchNormalization())
    m.add(layers.Dropout(DROPOUT))
    m.add(layers.Dense(32, activation="relu"))
    m.add(layers.Dropout(DROPOUT))
    m.add(layers.Dense(n_classes, activation="softmax"))
    m.compile(optimizer=keras.optimizers.Adam(LR),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def train_keras(model, X_tr, y_tr, X_va, y_va,
                epochs, batch, patience, seed,
                class_weight: dict | None = None):
    tf.random.set_seed(seed)
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                    patience=patience//2, min_lr=1e-6, verbose=0),
    ]
    t0 = time.time()
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va),
                     epochs=epochs, batch_size=batch,
                     callbacks=cbs, class_weight=class_weight, verbose=0)
    train_secs = time.time() - t0
    return model, hist, train_secs


def train_rf(X_tr, y_tr, seed):
    clf = RandomForestClassifier(
        n_estimators=RF_TREES,
        n_jobs=-1,
        random_state=seed,
        class_weight="balanced",
    )
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    return clf, time.time() - t0


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

def measure_latency(predict_fn, X_sample):
    """Single-sample inference latency (ms), averaged over LATENCY_TRIALS."""
    for _ in range(LATENCY_WARMUP):
        predict_fn(X_sample)
    t0 = time.perf_counter()
    for _ in range(LATENCY_TRIALS):
        predict_fn(X_sample)
    return (time.perf_counter() - t0) / LATENCY_TRIALS * 1000.0


def evaluate_predictions(y_true, y_pred, le_classes):
    """Return dict with accuracy, macro_f1, per_class_f1, confusion."""
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(le_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(len(le_classes)))
    return {
        "accuracy":      float(acc),
        "macro_f1":      float(macro_f1),
        "per_class_f1":  f1.tolist(),
        "per_class_sup": sup.tolist(),
        "confusion":     cm.tolist(),
    }


def rf_model_size_kb(rf: RandomForestClassifier) -> float:
    """Serialized model size in KB."""
    blob = pickle.dumps(rf)
    return len(blob) / 1024.0


def keras_model_size_kb(model: keras.Model) -> float:
    total = sum(np.prod(v.shape) for v in model.trainable_variables)
    return float(total * 4) / 1024.0   # 4 bytes per float32


# ═════════════════════════════════════════════════════════════════════════════
# REPORTING
# ═════════════════════════════════════════════════════════════════════════════

def plot_confusion(cm: np.ndarray, le_classes, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 9))
    norm = cm.astype(np.float32) / (cm.sum(axis=1, keepdims=True) + 1e-9)
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    names = [DISPLAY_NAME.get(int(c), str(c)) for c in le_classes]
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140); plt.close()


def plot_accuracy_bars(summary: dict, out_path: Path):
    names = list(summary.keys())
    means = [summary[n]["accuracy_mean"]*100 for n in names]
    stds  = [summary[n]["accuracy_std"]*100  for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    bars = ax.bar(names, means, yerr=stds, capsize=8,
                  color=colors[:len(names)], alpha=0.85, edgecolor="black")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.5, f"{m:.2f}%",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(f"Test Accuracy across {N_RUNS} runs (error bars = ±1 std)")
    ax.set_ylim(0, 105); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def plot_latency_bars(summary: dict, out_path: Path):
    names = list(summary.keys())
    means = [summary[n]["latency_ms_mean"] for n in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#3b82f6", "#10b981", "#f59e0b"]
    bars = ax.bar(names, means, color=colors[:len(names)],
                  alpha=0.85, edgecolor="black")
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m*1.05, f"{m:.2f} ms",
                ha="center", fontsize=10, fontweight="bold")
    ax.axhline(33.3, color="red", linestyle="--", linewidth=1,
               label="30 fps budget (33.3 ms)")
    ax.set_yscale("log")
    ax.set_ylabel("Inference latency per sample (ms, log scale)")
    ax.set_title("Inference Latency (single sample, CPU)")
    ax.legend(); ax.grid(axis="y", alpha=0.3, which="both")
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def plot_training_curves(curves: dict, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, c in curves.items():
        axes[0].plot(c["loss"],     label=f"{name} train")
        axes[0].plot(c["val_loss"], label=f"{name} val", linestyle="--")
        axes[1].plot(c["accuracy"],     label=f"{name} train")
        axes[1].plot(c["val_accuracy"], label=f"{name} val", linestyle="--")
    axes[0].set_title("Loss");     axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(out_path, dpi=140); plt.close()


def write_report(summary, per_class_f1_table, le_classes, out_path: Path):
    lines = []
    lines.append("# Model Comparison — Semaphore Translation\n")
    lines.append(f"_Generated by `compare_models.py`. {N_RUNS} runs per model._\n\n")

    lines.append("## TL;DR\n\n")
    rows = sorted(summary.items(), key=lambda kv: -kv[1]["accuracy_mean"])
    best = rows[0][0]
    lines.append(f"- **Best test accuracy**: `{best}` "
                 f"({summary[best]['accuracy_mean']*100:.2f}% "
                 f"± {summary[best]['accuracy_std']*100:.2f})\n")
    fastest = min(summary.items(), key=lambda kv: kv[1]["latency_ms_mean"])[0]
    lines.append(f"- **Fastest inference**: `{fastest}` "
                 f"({summary[fastest]['latency_ms_mean']:.2f} ms / sample)\n")
    smallest = min(summary.items(), key=lambda kv: kv[1]["model_size_kb_mean"])[0]
    lines.append(f"- **Smallest model**: `{smallest}` "
                 f"({summary[smallest]['model_size_kb_mean']:.1f} KB)\n\n")

    lines.append("## Methodology\n\n")
    lines.append("Three classifiers are trained and evaluated under a matched, "
                 "group-aware partition of the source frames. The group is the "
                 "`source_id` (original raw-frame index) — identical to the "
                 "strategy in `train_model.py`. The same source frames go to "
                 "the same side of the partition for all three models per run, "
                 "so any accuracy differences are attributable to the model "
                 "itself, not to partition luck or augmentation leak. All three "
                 "models use `class_weight='balanced'` to handle the ~2.6× "
                 "imbalance between the most and least frequent letters.\n\n")
    lines.append(f"- **DNN** — 24-dim per-frame features → `{'-'.join(map(str, DNN_HIDDEN))}` MLP "
                 f"with BatchNorm + Dropout({DROPOUT}). Trained on `poses_augmented.csv`.\n")
    lines.append(f"- **Random Forest** — 24-dim per-frame features → "
                 f"{RF_TREES} trees, `class_weight='balanced'`. "
                 f"Trained on `poses_augmented.csv`.\n")
    lines.append(f"- **LSTM** — sequences of length {WINDOW_LEN} (= {WINDOW_LEN/10:.1f}s at 10 fps) → "
                 f"`LSTM({LSTM_UNITS})` → `Dense(32)` → softmax. "
                 f"Sequences extracted from `poses_raw.csv` and augmented with "
                 f"the same primitives used for the DNN/RF data. Per-frame joint "
                 f"noise is drawn independently within each window — the LSTM "
                 f"has to learn to smooth it. Each window's group is the "
                 f"`source_id` of its center frame.\n\n")
    lines.append(f"- **Runs**: {N_RUNS} different group-split seeds. Numbers below are mean ± std.\n")
    lines.append(f"- **Latency**: single-sample wall-clock on CPU, averaged over "
                 f"{LATENCY_TRIALS} calls after {LATENCY_WARMUP} warmup calls.\n\n")

    lines.append("### Caveat on LSTM splitting\n\n")
    lines.append("Each LSTM window's group is the `source_id` of its CENTER frame, "
                 "but a window of length 10 spans 10 source IDs and the next "
                 f"window (stride {WINDOW_STRIDE}) shares {WINDOW_LEN-WINDOW_STRIDE} of them. "
                 "So two windows can land on opposite sides of the split while "
                 "sharing most of their underlying frames. This means LSTM test "
                 "accuracy may be modestly inflated by frame-level overlap "
                 "between adjacent train/test windows. Per-frame independent "
                 "joint noise in the augmentation provides real diversity that "
                 "mitigates this, but the LSTM number should be read as a "
                 "slight upper bound. A stricter alternative would group windows "
                 "by entire video, but with only 27 videos that would put whole "
                 "letters into test and make per-class numbers meaningless.\n\n")

    lines.append("## Summary\n\n")
    lines.append("| Model | Test acc (%) | Macro F1 | Train time (s) | Latency (ms) | Model size (KB) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for name in summary:
        s = summary[name]
        lines.append(
            f"| {name} | "
            f"{s['accuracy_mean']*100:.2f} ± {s['accuracy_std']*100:.2f} | "
            f"{s['macro_f1_mean']:.3f} ± {s['macro_f1_std']:.3f} | "
            f"{s['train_secs_mean']:.1f} ± {s['train_secs_std']:.1f} | "
            f"{s['latency_ms_mean']:.2f} ± {s['latency_ms_std']:.2f} | "
            f"{s['model_size_kb_mean']:.1f} ± {s['model_size_kb_std']:.1f} |\n"
        )
    lines.append("\n")

    lines.append("## Per-class F1 (mean across runs)\n\n")
    names = [DISPLAY_NAME.get(int(c), str(c)) for c in le_classes]
    header = "| Class | " + " | ".join(per_class_f1_table.keys()) + " |\n"
    sep    = "|---" * (1 + len(per_class_f1_table)) + "|\n"
    lines.append(header); lines.append(sep)
    for i, n in enumerate(names):
        row_vals = [f"{per_class_f1_table[m][i]:.3f}" for m in per_class_f1_table]
        lines.append(f"| {n} | " + " | ".join(row_vals) + " |\n")
    lines.append("\n")

    lines.append("## Discussion\n\n")
    lines.append("### Q1 — Is a DNN justified vs Random Forest on 24-dim features?\n\n")
    rf_acc  = summary.get("RandomForest", {}).get("accuracy_mean", float("nan"))
    dnn_acc = summary.get("DNN",          {}).get("accuracy_mean", float("nan"))
    rf_lat  = summary.get("RandomForest", {}).get("latency_ms_mean", float("nan"))
    dnn_lat = summary.get("DNN",          {}).get("latency_ms_mean", float("nan"))
    delta   = (dnn_acc - rf_acc) * 100
    lines.append(f"DNN scored {dnn_acc*100:.2f}% vs Random Forest {rf_acc*100:.2f}% — "
                 f"a gap of **{delta:+.2f}** percentage points. ")
    lines.append(f"At inference, DNN takes {dnn_lat:.2f} ms vs RF's {rf_lat:.2f} ms per sample. ")
    lines.append("Interpret in light of the std bands above: if the gap is smaller than "
                 "the sum of the two stds, the difference is not statistically meaningful "
                 "at this sample size.\n\n")

    lines.append("### Q2 — Native temporal (LSTM) vs static DNN + smoothing?\n\n")
    lstm_acc = summary.get("LSTM", {}).get("accuracy_mean", float("nan"))
    lstm_lat = summary.get("LSTM", {}).get("latency_ms_mean", float("nan"))
    delta2   = (lstm_acc - dnn_acc) * 100
    lines.append(f"LSTM scored {lstm_acc*100:.2f}% on whole-window classification vs "
                 f"DNN's per-frame {dnn_acc*100:.2f}% (gap **{delta2:+.2f}** pp). ")
    lines.append(f"LSTM latency is {lstm_lat:.2f} ms/window vs DNN's {dnn_lat:.2f} ms/frame.\n\n")
    lines.append("A key caveat: the deployed DNN already has temporal smoothing at "
                 "inference (mode over a 180-frame window, COMMIT_AFTER=30). The fairest "
                 "live-equivalent comparison would aggregate DNN per-frame predictions "
                 "across a window of the same length the LSTM sees, then compare to "
                 "LSTM window predictions. That extension is left as future work. Also "
                 "see the windowing caveat above — the LSTM number is a slight "
                 "upper bound due to adjacent-window frame overlap.\n\n")

    lines.append("## Files\n\n")
    lines.append("- `metrics.json` — raw numbers for every run\n")
    lines.append("- `metrics_summary.csv` — per-model summary\n")
    lines.append("- `per_class_f1.csv` — per-class F1 (mean across runs)\n")
    lines.append("- `accuracy_comparison.png`, `latency_comparison.png`, `training_curves.png`\n")
    lines.append("- `confusion_DNN.png`, `confusion_RandomForest.png`, `confusion_LSTM.png`\n")

    out_path.write_text("".join(lines), encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("  MODEL COMPARISON — DNN vs RandomForest vs LSTM  (v2, 24 dims)")
    print("=" * 60)

    print("[LOAD] Per-frame data (DNN, RF)…")
    X_frame, y_frame_raw, src_id_frame = load_per_frame()
    print(f"       X={X_frame.shape}  y={y_frame_raw.shape}  "
          f"unique sources={len(np.unique(src_id_frame))}")

    print("[LOAD] Temporal windows (LSTM)…")
    X_seq, y_seq_raw, src_id_seq = load_temporal()
    print(f"       X={X_seq.shape}  y={y_seq_raw.shape}  "
          f"unique groups={len(np.unique(src_id_seq))}")

    le = LabelEncoder().fit(np.concatenate([y_frame_raw, y_seq_raw]))
    y_frame = le.transform(y_frame_raw)
    y_seq   = le.transform(y_seq_raw)
    n_classes = len(le.classes_)
    print(f"[INFO] n_classes = {n_classes}")

    runs = {"DNN": [], "RandomForest": [], "LSTM": []}
    curves_first_run = {}

    for run in range(N_RUNS):
        seed = 42 + run
        print(f"\n──── RUN {run+1}/{N_RUNS}  (seed={seed}) ────────────────")

        # Matched split — union of source_ids across both datasets.
        # Same source frames land on the same side of the partition for all 3 models.
        all_groups = np.unique(np.concatenate([src_id_frame, src_id_seq]))
        train_g, val_g, test_g = matched_split(all_groups, seed)
        print(f"  groups  train={len(train_g)}  val={len(val_g)}  test={len(test_g)}")

        # ── Per-frame data partitions
        tr_f, va_f, te_f = apply_split(src_id_frame, train_g, val_g, test_g)
        scaler_f = StandardScaler().fit(X_frame[tr_f])
        Xtr_f = scaler_f.transform(X_frame[tr_f])
        Xva_f = scaler_f.transform(X_frame[va_f])
        Xte_f = scaler_f.transform(X_frame[te_f])
        ytr_f, yva_f, yte_f = y_frame[tr_f], y_frame[va_f], y_frame[te_f]
        cw_f = balanced_class_weight(ytr_f)
        print(f"  frames  train={len(ytr_f)} val={len(yva_f)} test={len(yte_f)}  "
              f"cw min={min(cw_f.values()):.2f}/max={max(cw_f.values()):.2f}")

        # ── Temporal window partitions
        tr_s, va_s, te_s = apply_split(src_id_seq, train_g, val_g, test_g)
        T, D = X_seq.shape[1], X_seq.shape[2]
        scaler_s = StandardScaler().fit(X_seq[tr_s].reshape(-1, D))
        Xtr_s = scaler_s.transform(X_seq[tr_s].reshape(-1, D)).reshape(-1, T, D)
        Xva_s = scaler_s.transform(X_seq[va_s].reshape(-1, D)).reshape(-1, T, D)
        Xte_s = scaler_s.transform(X_seq[te_s].reshape(-1, D)).reshape(-1, T, D)
        ytr_s, yva_s, yte_s = y_seq[tr_s], y_seq[va_s], y_seq[te_s]
        cw_s = balanced_class_weight(ytr_s)
        print(f"  windows train={len(ytr_s)} val={len(yva_s)} test={len(yte_s)}  "
              f"cw min={min(cw_s.values()):.2f}/max={max(cw_s.values()):.2f}")

        # ── DNN  (class-weighted, matching train_model.py)
        print("  [DNN] training…", flush=True)
        dnn = build_dnn(D, n_classes)
        dnn, h_dnn, t_dnn = train_keras(
            dnn, Xtr_f, ytr_f, Xva_f, yva_f,
            DNN_EPOCHS, DNN_BATCH, DNN_PATIENCE, seed,
            class_weight=cw_f,
        )
        y_pred = np.argmax(dnn.predict(Xte_f, verbose=0), axis=1)
        ev = evaluate_predictions(yte_f, y_pred, le.classes_)
        lat = measure_latency(
            lambda x: dnn(x, training=False).numpy(),
            Xte_f[:1].astype(np.float32)
        )
        ev.update(train_secs=t_dnn, latency_ms=lat,
                  model_size_kb=keras_model_size_kb(dnn))
        runs["DNN"].append(ev)
        print(f"        acc={ev['accuracy']*100:.2f}%  "
              f"macroF1={ev['macro_f1']:.3f}  lat={lat:.2f}ms")

        # ── Random Forest
        print("  [RF ] training…", flush=True)
        rf, t_rf = train_rf(Xtr_f, ytr_f, seed)
        y_pred = rf.predict(Xte_f)
        ev = evaluate_predictions(yte_f, y_pred, le.classes_)
        lat = measure_latency(rf.predict, Xte_f[:1])
        ev.update(train_secs=t_rf, latency_ms=lat,
                  model_size_kb=rf_model_size_kb(rf))
        runs["RandomForest"].append(ev)
        print(f"        acc={ev['accuracy']*100:.2f}%  "
              f"macroF1={ev['macro_f1']:.3f}  lat={lat:.2f}ms")

        # ── LSTM  (class-weighted)
        print("  [LSTM] training…", flush=True)
        lstm = build_lstm(T, D, n_classes)
        lstm, h_lstm, t_lstm = train_keras(
            lstm, Xtr_s, ytr_s, Xva_s, yva_s,
            LSTM_EPOCHS, LSTM_BATCH, LSTM_PATIENCE, seed,
            class_weight=cw_s,
        )
        y_pred = np.argmax(lstm.predict(Xte_s, verbose=0), axis=1)
        ev = evaluate_predictions(yte_s, y_pred, le.classes_)
        lat = measure_latency(
            lambda x: lstm(x, training=False).numpy(),
            Xte_s[:1].astype(np.float32)
        )
        ev.update(train_secs=t_lstm, latency_ms=lat,
                  model_size_kb=keras_model_size_kb(lstm))
        runs["LSTM"].append(ev)
        print(f"        acc={ev['accuracy']*100:.2f}%  "
              f"macroF1={ev['macro_f1']:.3f}  lat={lat:.2f}ms")

        # Save training curves from the first run for plotting
        if run == 0:
            curves_first_run["DNN"]  = {k: list(map(float, v)) for k, v in h_dnn.history.items()}
            curves_first_run["LSTM"] = {k: list(map(float, v)) for k, v in h_lstm.history.items()}

    # ── Aggregate
    summary = {}
    per_class_f1_table = {}
    avg_cm = {}
    for name, evs in runs.items():
        acc = np.array([e["accuracy"]     for e in evs])
        f1m = np.array([e["macro_f1"]     for e in evs])
        tt  = np.array([e["train_secs"]   for e in evs])
        lat = np.array([e["latency_ms"]   for e in evs])
        sz  = np.array([e["model_size_kb"] for e in evs])
        summary[name] = {
            "accuracy_mean":      acc.mean(), "accuracy_std":      acc.std(),
            "macro_f1_mean":      f1m.mean(), "macro_f1_std":      f1m.std(),
            "train_secs_mean":    tt.mean(),  "train_secs_std":    tt.std(),
            "latency_ms_mean":    lat.mean(), "latency_ms_std":    lat.std(),
            "model_size_kb_mean": sz.mean(),  "model_size_kb_std": sz.std(),
        }
        per_class_f1_table[name] = np.mean(
            [np.array(e["per_class_f1"]) for e in evs], axis=0
        ).tolist()
        avg_cm[name] = np.mean(
            [np.array(e["confusion"]) for e in evs], axis=0
        )

    # ── Persist artefacts
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"summary": summary,
                   "per_class_f1": per_class_f1_table,
                   "all_runs": {k: [{kk: vv for kk, vv in e.items() if kk != "confusion"}
                                    for e in v] for k, v in runs.items()},
                   "config": {
                       "n_runs": N_RUNS, "window_len": WINDOW_LEN,
                       "lstm_units": LSTM_UNITS, "rf_trees": RF_TREES,
                       "dnn_hidden": DNN_HIDDEN,
                   }}, f, indent=2)

    pd.DataFrame(summary).T.to_csv(OUT_DIR / "metrics_summary.csv")

    f1_df = pd.DataFrame(per_class_f1_table,
                         index=[DISPLAY_NAME.get(int(c), str(c)) for c in le.classes_])
    f1_df.to_csv(OUT_DIR / "per_class_f1.csv")

    for name, cm in avg_cm.items():
        plot_confusion(cm, le.classes_,
                       f"Confusion — {name} (mean of {N_RUNS} runs, row-normalised)",
                       OUT_DIR / f"confusion_{name}.png")
    plot_accuracy_bars(summary,  OUT_DIR / "accuracy_comparison.png")
    plot_latency_bars(summary,   OUT_DIR / "latency_comparison.png")
    plot_training_curves(curves_first_run, OUT_DIR / "training_curves.png")

    write_report(summary, per_class_f1_table, le.classes_, OUT_DIR / "report.md")

    print("\n" + "=" * 60)
    print("  DONE  —  outputs in ./model_comparison/")
    print("=" * 60)
    print("  • report.md         (open this first)")
    print("  • metrics.json      (raw numbers)")
    print("  • metrics_summary.csv")
    print("  • per_class_f1.csv")
    print("  • *.png             (plots)")


if __name__ == "__main__":
    import sys, traceback
    try:
        main()
    except MemoryError:
        print("\n[FATAL] MemoryError — try lowering LSTM_AUGS or N_RUNS.",
              file=sys.stderr, flush=True)
        sys.exit(2)
    except Exception:
        print("\n[FATAL] Unhandled exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)