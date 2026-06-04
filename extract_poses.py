import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
from pathlib import Path
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# ─── Configuration ────────────────────────────────────────────────────────────
VIDEOS_DIR      = Path("dataset")
OUTPUT_CSV      = Path("poses_raw.csv")
POSE_MODEL_PATH = Path("pose_landmarker.task")
SAMPLE_FPS      = 10 
VIDEO_EXTS      = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

LABEL_MAP: dict[str, int] = {
    **{chr(ord("A") + i): i + 1 for i in range(26)},
    **{str(i + 1):        i + 1 for i in range(26)},
    "SPACE": 27,
}
DISPLAY_NAME: dict[int, str] = {
    **{i + 1: chr(ord("A") + i) for i in range(26)},
    27: "SPC",
}
N_CLASSES = 27

LANDMARK_MAP = {
    "left_shoulder" : 11,
    "right_shoulder": 12,
    "left_elbow"    : 13,
    "right_elbow"   : 14,
    "left_wrist"    : 15,
    "right_wrist"   : 16,
}

COORD_COLS = [
    f"{name}_{axis}"
    for name in LANDMARK_MAP.keys()
    for axis in ("x", "y", "z")
]
ANGLE_COLS = [
    "left_upper_arm_angle",
    "right_upper_arm_angle",
    "left_forearm_angle",
    "right_forearm_angle",
    "left_elbow_bend",
    "right_elbow_bend",
]
FEATURE_COLS = COORD_COLS + ANGLE_COLS
LABEL_COL    = "label"

# ─── Model Download ───────────────────────────────────────────────────────────
def ensure_model_downloaded() -> None:
    if POSE_MODEL_PATH.exists():
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/"
        "pose_landmarker_full.task"
    )
    print(f"[INFO] Downloading pose model → {POSE_MODEL_PATH}")
    urllib.request.urlretrieve(url, POSE_MODEL_PATH)
    print("[✓] Model saved.")

# ─── Landmarker ───────────────────────────────────────────────────────────────
def build_landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH)
        ),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.60,
        min_pose_presence_confidence=0.60,
        min_tracking_confidence=0.60,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)

# ─── Feature Computation ──────────────────────────────────────────────────────
def _angle_vs_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(p2[0] - p1[0], p2[1] - p1[1])))

def _elbow_bend(shoulder: np.ndarray,
                elbow:    np.ndarray,
                wrist:    np.ndarray) -> float:
    """Interior angle at the elbow. Range: 0°–180°."""
    v1    = shoulder - elbow
    v2    = wrist    - elbow
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t, -1.0, 1.0))))

def _normalise(lm_raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    ls  = lm_raw["left_shoulder"]
    rs  = lm_raw["right_shoulder"]
    mid = (ls + rs) / 2.0
    sw  = np.linalg.norm(ls[:2] - rs[:2]) + 1e-8
    return {name: (xyz - mid) / sw for name, xyz in lm_raw.items()}

def build_feature_vector(raw_lms) -> np.ndarray | None:
    lm_raw: dict[str, np.ndarray] = {
        name: np.array([raw_lms[idx].x,
                        raw_lms[idx].y,
                        raw_lms[idx].z], dtype=np.float32)
        for name, idx in LANDMARK_MAP.items()
    }

    sw = np.linalg.norm(
        lm_raw["left_shoulder"][:2] - lm_raw["right_shoulder"][:2]
    )
    if sw < 0.05:
        return None

    lm_norm = _normalise(lm_raw)
    coords  = [v for name in LANDMARK_MAP for v in lm_norm[name].tolist()]

    ls2 = lm_raw["left_shoulder"][:2]
    rs2 = lm_raw["right_shoulder"][:2]
    le2 = lm_raw["left_elbow"][:2]
    re2 = lm_raw["right_elbow"][:2]
    lw2 = lm_raw["left_wrist"][:2]
    rw2 = lm_raw["right_wrist"][:2]

    angles = [
        _angle_vs_vertical(ls2, le2),
        _angle_vs_vertical(rs2, re2),
        _angle_vs_vertical(le2, lw2),
        _angle_vs_vertical(re2, rw2),
        _elbow_bend(ls2, le2, lw2),
        _elbow_bend(rs2, re2, rw2),
    ]

    return np.array(coords + angles, dtype=np.float32)

# ─── Video Discovery ─────────────────────────────────────────────────────────
def discover_videos(videos_dir: Path) -> list[tuple[Path, int]]:
    """
    Scan videos_dir for recognised video files.

    Accepted stems (case-insensitive):
      A–Z       → labels 1–26
      1–26      → labels 1–26
      Space     → label 27

    Returns list of (path, label) sorted by label.
    """
    entries: list[tuple[Path, int]] = []
    unrecognised: list[str] = []

    for f in videos_dir.iterdir():
        if f.suffix.lower() not in VIDEO_EXTS:
            continue
        stem  = f.stem.strip().upper()
        label = LABEL_MAP.get(stem)
        if label is not None:
            entries.append((f, label))
        else:
            unrecognised.append(f.name)

    if unrecognised:
        print(f"[WARN] Unrecognised files (skipped): {unrecognised}")

    if not entries:
        raise FileNotFoundError(
            f"No recognised video files found in '{videos_dir}'.\n"
            "Name them A.mp4–Z.mp4, Space.mp4  (or 1.mp4–26.mp4)."
        )

    entries.sort(key=lambda x: x[1])
    return entries

# ─── Per-Video Extraction ────────────────────────────────────────────────────
def extract_from_video(video_path: Path,
                       label: int,
                       landmarker: PoseLandmarker,
                       sample_fps: float) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open '{video_path}' — skipping.")
        return []

    native_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / native_fps
    frame_step   = max(1, int(round(native_fps / sample_fps)))

    records: list[dict] = []
    frame_idx = ok = fail = 0

    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=image_rgb)
            result    = landmarker.detect(mp_image)

            if result.pose_landmarks:
                features = build_feature_vector(result.pose_landmarks[0])
                if features is not None:
                    rec = {col: float(v)
                           for col, v in zip(FEATURE_COLS, features)}
                    rec[LABEL_COL] = label
                    records.append(rec)
                    ok += 1
                else:
                    fail += 1
            else:
                fail += 1

        frame_idx += 1

    cap.release()

    name = DISPLAY_NAME.get(label, str(label))
    print(f"  [{label:>2}] {name:<4}  |  "
          f"dur: {duration_s:.1f}s  "
          f"sampled: {ok+fail}  "
          f"→  {ok} ok, {fail} skipped")
    return records

# ─── Main ─────────────────────────────────────────────────────────────────────
def extract_all(videos_dir: Path) -> pd.DataFrame:
    videos  = discover_videos(videos_dir)
    records: list[dict] = []

    print(f"[INFO] Found {len(videos)} video(s)  (expecting up to {N_CLASSES})")
    print(f"[INFO] Sampling at {SAMPLE_FPS} fps\n")

    with build_landmarker() as landmarker:
        for video_path, label in videos:
            records.extend(
                extract_from_video(video_path, label, landmarker, SAMPLE_FPS)
            )

    return pd.DataFrame(records, columns=FEATURE_COLS + [LABEL_COL])


def main() -> None:
    if not VIDEOS_DIR.exists():
        raise FileNotFoundError(
            f"'{VIDEOS_DIR}' not found.\n"
            "Create a 'videos/' folder with one video per letter/space."
        )

    ensure_model_downloaded()

    print("=" * 60)
    print("  STEP 1 — Video Pose Extraction  (27 classes, 24 dims)")
    print("=" * 60)

    df = extract_all(VIDEOS_DIR)

    print(f"\n[INFO] Total samples : {len(df)}")
    print(f"[INFO] Feature dims  : {len(FEATURE_COLS)}")
    print(f"[INFO] Per-class counts:")
    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        name = DISPLAY_NAME.get(int(lbl), str(lbl))
        bar  = "█" * (cnt // 5)
        print(f"  [{int(lbl):>2}] {name:<4}  {cnt:>4}  {bar}")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[✓] Saved → '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()