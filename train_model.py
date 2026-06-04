import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_CSV    = Path("poses_augmented.csv")
MODEL_OUT    = Path("semaphore_model.keras")
SCALER_OUT   = Path("scaler.pkl")
ENCODER_OUT  = Path("label_encoder.npy")
HISTORY_PLOT = Path("training_history.png")

EPOCHS      = 100
BATCH_SIZE  = 64
LR          = 1e-3
PATIENCE    = 15
SEED        = 42
N_CLASSES   = 27
DROPOUT     = 0.20
HIDDEN      = [128, 64, 32]
LABEL_COL   = "label"
SOURCE_COL  = "source_id"

DISPLAY_NAME: dict[int, str] = {
    **{i+1: chr(ord("A")+i) for i in range(26)},
    27: "SPC",
}

# ─── Data ─────────────────────────────────────────────────────────────────────

def load_and_split(csv_path: Path):
    df = pd.read_csv(csv_path)
    has_groups = SOURCE_COL in df.columns

    # Feature columns = everything except label and source_id
    feature_cols = [c for c in df.columns if c not in (LABEL_COL, SOURCE_COL)]
    print(f"[INFO] Feature dims  : {len(feature_cols)}")
    print(f"[INFO] Classes found : {sorted(df[LABEL_COL].unique())}")

    X      = df[feature_cols].values.astype(np.float32)
    y_raw  = df[LABEL_COL].values.astype(int)
    groups = df[SOURCE_COL].values if has_groups else None

    le     = LabelEncoder()
    y      = le.fit_transform(y_raw)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    if has_groups:
        # Group-aware split: all augmented copies of one source frame
        # stay on the same side of every split boundary.  No leak.
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
        tv_idx, test_idx = next(gss1.split(X, y, groups))
        X_tv, X_test     = X[tv_idx],   X[test_idx]
        y_tv, y_test     = y[tv_idx],   y[test_idx]
        groups_tv        = groups[tv_idx]

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.20/0.90, random_state=SEED)
        tr_idx, val_idx = next(gss2.split(X_tv, y_tv, groups_tv))
        X_train, X_val  = X_tv[tr_idx], X_tv[val_idx]
        y_train, y_val  = y_tv[tr_idx], y_tv[val_idx]
        print("[INFO] Split mode    : GROUP-AWARE (no aug leak)")
    else:
        # Fallback for older CSVs without source_id — leaks via aug
        print("[WARN] No source_id column — falling back to stratified split.")
        print("[WARN] Test accuracy will be inflated by augmentation leak.")
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.10, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=(0.20/0.90), random_state=SEED, stratify=y_tv
        )

    print(f"  Train      : {len(X_train):>6}  (~70%)")
    print(f"  Validation : {len(X_val):>6}  (~20%)")
    print(f"  Test       : {len(X_test):>6}  (~10%)")
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, le

# ─── Model ────────────────────────────────────────────────────────────────────

def build_model(input_dim: int, n_classes: int) -> keras.Model:
    model = keras.Sequential(name="SemaphoreDNN")
    model.add(layers.Input(shape=(input_dim,), name="pose_vector"))
    for i, units in enumerate(HIDDEN, 1):
        model.add(layers.Dense(units,           name=f"dense_{i}"))
        model.add(layers.BatchNormalization(    name=f"bn_{i}"))
        model.add(layers.Activation("relu",     name=f"relu_{i}"))
        model.add(layers.Dropout(DROPOUT,       name=f"drop_{i}"))
    model.add(layers.Dense(n_classes, activation="softmax", name="output"))
    return model

# ─── Training ─────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, X_val, y_val):
    # Compute balanced class weights so M (132 samples) doesn't get
    # overwhelmed by C (346 samples) during gradient updates.
    cw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight_dict = {int(c): float(w) for c, w in zip(np.unique(y_train), cw)}
    print(f"[INFO] Class weights : min={cw.min():.2f}  max={cw.max():.2f}  "
          f"ratio={cw.max()/cw.min():.2f}x")

    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    cbs = [
        callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=PATIENCE//2, min_lr=1e-6, verbose=1,
        ),
        callbacks.ModelCheckpoint(
            str(MODEL_OUT), monitor="val_accuracy",
            save_best_only=True, verbose=1,
        ),
    ]
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=cbs, class_weight=class_weight_dict, verbose=1,
    )

# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, le):
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[TEST] Loss: {loss:.4f}   Accuracy: {acc*100:.2f}%")
    print("       (Group-aware split — this number reflects real generalization.)")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f"\n  {'Lbl':>4}  {'Name':>5}  {'Correct':>8}  {'Total':>6}  {'Acc%':>6}")
    print("  " + "─"*38)
    for idx in range(len(le.classes_)):
        mask  = y_test == idx
        total = mask.sum()
        if total == 0:
            continue
        correct = (y_pred[mask] == idx).sum()
        raw_lbl = int(le.classes_[idx])
        name    = DISPLAY_NAME.get(raw_lbl, str(raw_lbl))
        print(f"  [{raw_lbl:>2}]   {name:<5}  "
              f"{correct:>7}   {total:>5}   {correct/total*100:>5.1f}%")

    # Also print top confusion pairs — useful for iterating
    print("\n  Top confusion pairs (true → predicted):")
    confusions: dict[tuple[int,int], int] = {}
    for t, p in zip(y_test, y_pred):
        if t != p:
            confusions[(int(t), int(p))] = confusions.get((int(t), int(p)), 0) + 1
    for (t, p), n in sorted(confusions.items(), key=lambda x: -x[1])[:10]:
        t_name = DISPLAY_NAME.get(int(le.classes_[t]), "?")
        p_name = DISPLAY_NAME.get(int(le.classes_[p]), "?")
        print(f"    {t_name:>3} → {p_name:<3}  ({n}×)")

def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Training History — Semaphore DNN v5 (group-aware split)")
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy"); axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss"); axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(HISTORY_PLOT), dpi=150)
    print(f"[✓] Plot → '{HISTORY_PLOT}'")

# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    tf.random.set_seed(SEED)
    np.random.seed(SEED)
    print("=" * 60)
    print("  STEP 3 — DNN Training  (v5, group-aware split + class weights)")
    print("=" * 60)
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"'{INPUT_CSV}' not found — run augment_data.py first.")

    X_train, X_val, X_test, y_train, y_val, y_test, scaler, le = load_and_split(INPUT_CSV)

    n_classes = len(le.classes_)
    print(f"[INFO] n_classes     : {n_classes}")

    model   = build_model(X_train.shape[1], n_classes)
    history = train(model, X_train, y_train, X_val, y_val)
    evaluate(model, X_test, y_test, le)

    with open(SCALER_OUT, "wb") as f:
        pickle.dump(scaler, f)
    np.save(str(ENCODER_OUT), le.classes_)
    plot_history(history)

    print(f"\n[✓] Model  → '{MODEL_OUT}'")
    print(f"[✓] Scaler → '{SCALER_OUT}'")
    print(f"[✓] Labels → '{ENCODER_OUT}'")

if __name__ == "__main__":
    main()