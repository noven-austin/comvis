"""
============================================================
 STEP 4 — LIVE INFERENCE WITH SKELETON OVERLAY  (v6, 24 dims)
 Temporal Vision-Language Model | Semaphore Translation
============================================================
 Compatible with: mediapipe >= 0.10  (Tasks API)

 27 classes: A–Z + Space
 MIRROR = False — video plays as recorded (no flip)

 Controls:  Q / ESC = quit   C = clear   SPACE = pause

 Changes from v5:
   - 24-dim feature vector (lateral wrist features removed —
     they were duplicates of normalised wrist X-coords).
     Must match extract_poses.py exactly.
"""

import cv2
import numpy as np
import pickle
import mediapipe as mp
from pathlib import Path
from collections import deque
from scipy import stats
import tensorflow as tf
import argparse
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# ─── Configuration ────────────────────────────────────────────────────────────

MODEL_PATH      = Path("semaphore_model.keras")
SCALER_PATH     = Path("scaler.pkl")
ENCODER_PATH    = Path("label_encoder.npy")
POSE_MODEL_PATH = Path("pose_landmarker.task")

CONF_THRESHOLD  = 0.50
WINDOW_SIZE     = 180
MIN_STABLE      = 20
COMMIT_AFTER    = 30

# False = video plays exactly as recorded (correct for self-recorded videos)
# True  = flip horizontally (only needed if video appears mirrored)
MIRROR          = False

LANDMARK_MAP = {
    "left_shoulder" : 11,
    "right_shoulder": 12,
    "left_elbow"    : 13,
    "right_elbow"   : 14,
    "left_wrist"    : 15,
    "right_wrist"   : 16,
}

# Label → display string  (A–Z for 1–26, space character for 27)
DISPLAY_MAP: dict[int, str] = {
    **{i+1: chr(ord("A")+i) for i in range(26)},
    27: " ",        # Space outputs an actual space in the text
}

FONT = cv2.FONT_HERSHEY_SIMPLEX

# ─── Skeleton Drawing ─────────────────────────────────────────────────────────

SKELETON_CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
]

COL_BONE_L   = (0,   220,  80)
COL_BONE_R   = (0,   140, 255)
COL_BONE_MID = (200, 200, 200)
COL_JOINT    = (255, 255,   0)
COL_DROPPED  = (60,   60, 220)

def _bone_colour(a: str, b: str) -> tuple:
    if "left"  in a and "left"  in b: return COL_BONE_L
    if "right" in a and "right" in b: return COL_BONE_R
    return COL_BONE_MID

def landmarks_to_pixels(raw_lms, w: int, h: int) -> dict[str, tuple[int,int]]:
    return {
        name: (int(raw_lms[idx].x * w), int(raw_lms[idx].y * h))
        for name, idx in LANDMARK_MAP.items()
    }

def draw_skeleton(frame: np.ndarray,
                  px: dict[str, tuple[int,int]],
                  dropped: bool) -> np.ndarray:
    jcol = COL_DROPPED if dropped else COL_JOINT
    for a, b in SKELETON_CONNECTIONS:
        if a in px and b in px:
            cv2.line(frame, px[a], px[b], _bone_colour(a, b), 3, cv2.LINE_AA)
    for (x, y) in px.values():
        cv2.circle(frame, (x, y), 7, (0,0,0), -1, cv2.LINE_AA)
        cv2.circle(frame, (x, y), 5, jcol,    -1, cv2.LINE_AA)
    return frame

# ─── Feature Extraction — must stay in sync with extract_poses.py ────────────

def _angle_vs_vertical(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(p2[0]-p1[0], p2[1]-p1[1])))

def _elbow_bend(s, e, w) -> float:
    v1 = s-e; v2 = w-e
    cos_t = np.dot(v1,v2) / (np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
    return float(np.degrees(np.arccos(np.clip(cos_t,-1.0,1.0))))

def _normalise(lm_raw: dict) -> dict:
    ls  = lm_raw["left_shoulder"]
    rs  = lm_raw["right_shoulder"]
    mid = (ls + rs) / 2.0
    sw  = np.linalg.norm(ls[:2] - rs[:2]) + 1e-8
    return {name: (xyz - mid) / sw for name, xyz in lm_raw.items()}

def build_feature_vector(raw_lms) -> np.ndarray | None:
    """
    24-dim feature vector — identical layout to extract_poses.py:
      0–17 : normalised XYZ for 6 landmarks
      18–23: 6 joint angles
    """
    lm_raw = {
        name: np.array([raw_lms[idx].x, raw_lms[idx].y, raw_lms[idx].z],
                       dtype=np.float32)
        for name, idx in LANDMARK_MAP.items()
    }
    sw = np.linalg.norm(
        lm_raw["left_shoulder"][:2] - lm_raw["right_shoulder"][:2]
    )
    if sw < 0.05:
        return None

    lm_norm = _normalise(lm_raw)
    coords  = [v for name in LANDMARK_MAP for v in lm_norm[name].tolist()]

    ls2, rs2 = lm_raw["left_shoulder"][:2],  lm_raw["right_shoulder"][:2]
    le2, re2 = lm_raw["left_elbow"][:2],     lm_raw["right_elbow"][:2]
    lw2, rw2 = lm_raw["left_wrist"][:2],     lm_raw["right_wrist"][:2]

    angles = [
        _angle_vs_vertical(ls2, le2), _angle_vs_vertical(rs2, re2),
        _angle_vs_vertical(le2, lw2), _angle_vs_vertical(re2, rw2),
        _elbow_bend(ls2, le2, lw2),   _elbow_bend(rs2, re2, rw2),
    ]

    return np.array(coords + angles, dtype=np.float32)

# ─── MediaPipe Landmarker ─────────────────────────────────────────────────────

def build_video_landmarker() -> PoseLandmarker:
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(POSE_MODEL_PATH)
        ),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.55,
        min_pose_presence_confidence=0.55,
        min_tracking_confidence=0.55,
        output_segmentation_masks=False,
    )
    return PoseLandmarker.create_from_options(options)

# ─── Load Artefacts ───────────────────────────────────────────────────────────

def load_artefacts():
    for p in (MODEL_PATH, SCALER_PATH, ENCODER_PATH):
        if not p.exists():
            raise FileNotFoundError(f"'{p}' missing — run train_model.py first.")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    le_classes = np.load(str(ENCODER_PATH))
    print(f"[✓] Model     : {MODEL_PATH}")
    print(f"[✓] Scaler    : {SCALER_PATH}")
    print(f"[✓] Classes   : {le_classes}")
    return model, scaler, le_classes

# ─── Filter 1 — Confidence Thresholding ──────────────────────────────────────

def confidence_filter(probs, threshold):
    conf = float(np.max(probs))
    cls  = int(np.argmax(probs))
    return (None, conf) if conf < threshold else (cls, conf)

# ─── Filter 2 — Temporal Smoother ────────────────────────────────────────────

class TemporalSmoother:
    def __init__(self):
        self.window          = deque(maxlen=WINDOW_SIZE)
        self._stable_count   = 0
        self._last_mode      = None
        self._last_committed = None
        self.output_text     = ""

    def push(self, cls_idx: int) -> str | None:
        self.window.append(cls_idx)
        if len(self.window) < MIN_STABLE:
            return None

        res   = stats.mode(list(self.window), keepdims=False)
        mode  = int(res.mode)
        count = int(res.count)

        if count < MIN_STABLE:
            self._stable_count = 0
            self._last_mode    = None
            return None

        self._stable_count = (self._stable_count + 1
                              if mode == self._last_mode else 1)
        self._last_mode = mode

        if self._stable_count >= COMMIT_AFTER:
            self.window.clear()
            self._stable_count = 0
            self._last_mode    = None

            raw_lbl = mode + 1
            token   = DISPLAY_MAP.get(raw_lbl, "?")

            if token == " ":
                if self.output_text and self.output_text[-1] != " ":
                    self.output_text     += token
                    self._last_committed  = None
                    return "SPC"
                return None
            else:
                if self._last_committed != token:
                    self.output_text     += token
                    self._last_committed  = token
                    return token
                return None

        return None

    def clear(self):
        self.window.clear()
        self._stable_count   = 0
        self._last_mode      = None
        self._last_committed = None
        self.output_text     = ""

    @property
    def mode_label(self) -> int | None:
        """Return the current window mode as a 1-indexed label, or None."""
        if not self.window:
            return None
        return int(stats.mode(list(self.window), keepdims=False).mode) + 1

    @property
    def mode_display(self) -> str:
        lbl = self.mode_label
        if lbl is None:
            return "—"
        d = DISPLAY_MAP.get(lbl, "?")
        return "SPC" if d == " " else d

# ─── HUD ─────────────────────────────────────────────────────────────────────

def draw_hud(frame, smoother, cur_display, confidence, dropped, paused):
    h, w = frame.shape[:2]
    panel = frame.copy()
    cv2.rectangle(panel, (0, 0), (w, 195), (15, 15, 15), -1)
    cv2.addWeighted(panel, 0.68, frame, 0.32, 0, frame)

    status = "  [PAUSED]" if paused else ""
    cv2.putText(frame, f"SEMAPHORE TRANSLATOR{status}",
                (12, 28), FONT, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

    drop_tag   = "  [TRANSITION — dropped]" if dropped else ""
    conf_color = (60, 220, 60) if not dropped else (60, 60, 220)
    cv2.putText(frame,
                f"Current : {cur_display:<4}  conf: {confidence*100:.1f}%{drop_tag}",
                (12, 58), FONT, 0.60, conf_color, 1, cv2.LINE_AA)

    cv2.putText(frame,
                f"Mode    : {smoother.mode_display:<4}  "
                f"(win {len(smoother.window)}/{WINDOW_SIZE})",
                (12, 88), FONT, 0.60, (140, 200, 255), 1, cv2.LINE_AA)

    # Show output text — replace spaces with "·" so they're visible on screen
    visible_output = smoother.output_text.replace(" ", "·")
    display_text   = visible_output[-40:] if len(visible_output) > 40 \
                     else visible_output
    cv2.putText(frame, f"Output  : {display_text}",
                (12, 122), FONT, 0.72, (50, 230, 255), 2, cv2.LINE_AA)

    bx, by, bw, bh = 12, 148, 280, 14
    fill = int(bw * len(smoother.window) / WINDOW_SIZE)
    cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (70, 70, 70), -1)
    cv2.rectangle(frame, (bx, by), (bx+fill, by+bh), (50, 200, 120), -1)

    cv2.putText(frame, "[C] Clear   [Q/ESC] Quit   [SPACE key] Pause",
                (12, 182), FONT, 0.44, (150, 150, 150), 1, cv2.LINE_AA)
    return frame

# ─── Main Inference Loop ─────────────────────────────────────────────────────

def run_inference(source) -> None:
    print("=" * 60)
    print("  STEP 4 — Inference + Skeleton  (v6, 27 classes, 24 dims)")
    print("=" * 60)
    print(f"[INFO] Mirror flip : {'ON' if MIRROR else 'OFF'}")

    model, scaler, le_classes = load_artefacts()
    smoother  = TemporalSmoother()
    cap       = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {source}")

    native_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = 0

    print(f"[INFO] Stream      : {source}")
    print("[INFO] Controls    : Q/ESC=quit   C=clear   SPACE=pause\n")

    paused = False

    with build_video_landmarker() as landmarker:
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                smoother.clear()
                print("[INFO] Output buffer cleared.")
            if key == ord(" "):
                paused = not paused

            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of stream.")
                break

            frame_count  += 1
            timestamp_ms  = int(frame_count * 1000 / native_fps)

            if MIRROR:
                frame = cv2.flip(frame, 1)

            h, w = frame.shape[:2]

            cur_display = "—"
            confidence  = 0.0
            dropped     = False

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=image_rgb)
            result    = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                raw_lms   = result.pose_landmarks[0]
                px_coords = landmarks_to_pixels(raw_lms, w, h)

                if not paused:
                    features = build_feature_vector(raw_lms)
                    if features is not None:
                        X = scaler.transform(features.reshape(1, -1))
                        probs = model(X, training=False).numpy()[0]

                        pred_cls, confidence = confidence_filter(
                            probs, CONF_THRESHOLD
                        )

                        if pred_cls is None:
                            dropped = True
                            raw_lbl     = int(le_classes[int(np.argmax(probs))])
                            d           = DISPLAY_MAP.get(raw_lbl, "?")
                            cur_display = ("SPC?" if d == " " else d + "?")
                        else:
                            raw_lbl     = int(le_classes[pred_cls])
                            d           = DISPLAY_MAP.get(raw_lbl, "?")
                            cur_display = "SPC" if d == " " else d

                            committed = smoother.push(pred_cls)
                            if committed:
                                print(f"[COMMITTED] '{committed}'  →  "
                                      f"'{smoother.output_text}'")

                frame = draw_skeleton(frame, px_coords, dropped)

            frame = draw_hud(frame, smoother, cur_display,
                             confidence, dropped, paused)
            cv2.imshow("Semaphore Translator", frame)

    cap.release()
    cv2.destroyAllWindows()
    print(f"\n[✓] Final output: '{smoother.output_text}'")

# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semaphore Inference")
    parser.add_argument("--source", default="0",
                        help="Webcam index or video file path (default: 0)")
    args   = parser.parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run_inference(source)