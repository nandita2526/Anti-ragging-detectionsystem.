"""
Anti-Ragging Detection System
Rule-Based Computer Vision System for Real-Time Anti-Ragging Detection
in Institutional CCTV Surveillance using HOG+SVM and Centroid Tracking.

Based on paper: "A Rule-Based Computer Vision System for Real-Time
Anti-Ragging Detection in Institutional CCTV Surveillance"
"""

import cv2
import numpy as np
import csv
import time
from collections import OrderedDict
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION PARAMETERS
# ─────────────────────────────────────────────
GROUP_DISTANCE_THRESHOLD = 120   # θ_group  – pixels; ~1.2 m at typical corridor height
MIN_GROUP_SIZE           = 3     # N_min    – persons needed to flag a group
TIME_THRESHOLD           = 8.0   # τ        – seconds a group must persist before alert
SPEED_THRESHOLD          = 25    # v_max    – pixels/frame for fast-mover detection
MAX_DISAPPEARED          = 30    # D_max    – frames before a track is dropped
HOG_SCALE_FACTOR         = 1.05  # s        – image-pyramid scale for HOG detector
NMS_IOU_THRESHOLD        = 0.65  # IoU threshold for Non-Maximum Suppression
FRAME_WIDTH              = 960
FRAME_HEIGHT             = 540
LOG_FILE                 = "alert_log.csv"


# ─────────────────────────────────────────────
# CENTROID TRACKER
# ─────────────────────────────────────────────
class CentroidTracker:
    """
    Greedy centroid-based multi-object tracker.
    Assigns unique IDs to detected persons and tracks them across frames.
    Objects not matched for MAX_DISAPPEARED consecutive frames are removed.
    """

    def __init__(self, max_disappeared: int = MAX_DISAPPEARED):
        self.next_object_id = 0
        self.objects: OrderedDict[int, np.ndarray] = OrderedDict()   # id → centroid
        self.disappeared: OrderedDict[int, int]    = OrderedDict()   # id → frame count
        self.max_disappeared = max_disappeared

    def register(self, centroid: np.ndarray) -> None:
        self.objects[self.next_object_id]    = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id: int) -> None:
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects: list[tuple]) -> OrderedDict:
        """
        rects: list of (x, y, w, h) bounding boxes from the detector.
        Returns the current id→centroid mapping.
        """
        # No detections — increment disappeared counters
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        # Compute input centroids
        input_centroids = np.array(
            [(x + w // 2, y + h // 2) for (x, y, w, h) in rects],
            dtype="int"
        )

        # No existing objects — register all
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects

        # Greedy matching: find nearest existing object for each detection
        object_ids       = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        # Euclidean distance matrix: existing × input
        D = np.linalg.norm(
            np.array(object_centroids)[:, np.newaxis] -
            input_centroids[np.newaxis, :],
            axis=2
        )

        # Sort rows by minimum column value, then columns by sorted row index
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows: set[int] = set()
        used_cols: set[int] = set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            oid = object_ids[row]
            self.objects[oid]     = input_centroids[col]
            self.disappeared[oid] = 0
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            oid = object_ids[row]
            self.disappeared[oid] += 1
            if self.disappeared[oid] > self.max_disappeared:
                self.deregister(oid)

        # Register new detections
        unused_cols = set(range(D.shape[1])) - used_cols
        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects


# ─────────────────────────────────────────────
# HOG PERSON DETECTOR  (with NMS)
# ─────────────────────────────────────────────
def build_hog_detector() -> cv2.HOGDescriptor:
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return hog


def non_max_suppression(boxes: np.ndarray, iou_threshold: float) -> np.ndarray:
    """
    Suppress overlapping bounding boxes by IoU.
    boxes: (N, 4) array of [x, y, w, h].
    """
    if len(boxes) == 0:
        return boxes

    x1 = boxes[:, 0].astype(float)
    y1 = boxes[:, 1].astype(float)
    x2 = (boxes[:, 0] + boxes[:, 2]).astype(float)
    y2 = (boxes[:, 1] + boxes[:, 3]).astype(float)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs  = np.argsort(y2)          # sort by bottom edge
    picks = []

    while len(idxs) > 0:
        last = idxs[-1]
        picks.append(last)
        idxs = idxs[:-1]

        # Intersection
        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        intersection = w * h
        iou = intersection / (areas[last] + areas[idxs] - intersection)

        idxs = idxs[iou <= iou_threshold]

    return boxes[picks]


def detect_persons(hog: cv2.HOGDescriptor,
                   gray_frame: np.ndarray) -> list[tuple]:
    """
    Run HOG+SVM detector at multiple scales and return NMS-filtered boxes.
    Returns list of (x, y, w, h).
    """
    boxes, _ = hog.detectMultiScale(
        gray_frame,
        winStride=(8, 8),
        padding=(4, 4),
        scale=HOG_SCALE_FACTOR,
    )

    if len(boxes) == 0:
        return []

    boxes = np.array([[x, y, w, h] for (x, y, w, h) in boxes])
    boxes = non_max_suppression(boxes, NMS_IOU_THRESHOLD)
    return [tuple(b) for b in boxes]


# ─────────────────────────────────────────────
# RULE-BASED BEHAVIOURAL ANALYSIS
# ─────────────────────────────────────────────
class BehaviourAnalyser:
    """
    Evaluates the three ragging-detection rules:
      Rule 1 – Group Formation
      Rule 2 – Temporal Persistence
      Rule 3 – Motion Anomaly (fast mover in group)
    """

    def __init__(self):
        # group_key (frozenset of IDs) → first_seen timestamp
        self.group_timers: dict[frozenset, float] = {}
        # object_id → previous centroid
        self.prev_centroids: dict[int, np.ndarray] = {}

    def analyse(
        self,
        objects: OrderedDict,
        current_time: float,
    ) -> dict:
        """
        Returns a result dict:
          'victims'    : set of object IDs flagged as victims
          'suspects'   : set of object IDs in suspect groups
          'fast_movers': set of object IDs with speed > threshold
          'alert'      : bool — True if any rule fully triggered
          'groups'     : list of frozensets (one per detected cluster)
          'elapsed'    : dict group_key → elapsed seconds
        """
        result = {
            "victims":     set(),
            "suspects":    set(),
            "fast_movers": set(),
            "alert":       False,
            "groups":      [],
            "elapsed":     {},
        }

        if len(objects) < MIN_GROUP_SIZE + 1:
            # Not enough people in frame to trigger any rule
            self.prev_centroids = dict(objects)
            return result

        ids        = list(objects.keys())
        centroids  = np.array(list(objects.values()), dtype=float)

        # ── Rule 3: Motion Anomaly ─────────────────────────────────────
        fast_movers: set[int] = set()
        for oid, centroid in objects.items():
            if oid in self.prev_centroids:
                displacement = np.linalg.norm(centroid - self.prev_centroids[oid])
                if displacement > SPEED_THRESHOLD:
                    fast_movers.add(oid)

        result["fast_movers"] = fast_movers
        self.prev_centroids   = dict(objects)

        # ── Rule 1: Group Formation ────────────────────────────────────
        # Pairwise Euclidean distances
        dist_matrix = np.linalg.norm(
            centroids[:, np.newaxis] - centroids[np.newaxis, :], axis=2
        )

        active_group_keys: set[frozenset] = set()

        for i, victim_id in enumerate(ids):
            neighbors = [
                ids[j] for j in range(len(ids))
                if j != i and dist_matrix[i, j] < GROUP_DISTANCE_THRESHOLD
            ]
            if len(neighbors) >= MIN_GROUP_SIZE:
                result["victims"].add(victim_id)
                result["suspects"].update(neighbors)

                group_key = frozenset(neighbors + [victim_id])
                active_group_keys.add(group_key)

                if group_key not in result["groups"]:
                    result["groups"].append(group_key)

                # ── Rule 2: Temporal Persistence ──────────────────────
                if group_key not in self.group_timers:
                    self.group_timers[group_key] = current_time

                elapsed = current_time - self.group_timers[group_key]
                result["elapsed"][group_key] = elapsed

                if elapsed >= TIME_THRESHOLD:
                    result["alert"] = True

                # ── Rule 3 (combined with group): Immediate alert ─────
                if fast_movers & group_key:
                    result["alert"] = True

        # Clean up stale group timers
        stale = set(self.group_timers.keys()) - active_group_keys
        for key in stale:
            del self.group_timers[key]

        return result


# ─────────────────────────────────────────────
# VISUALISATION HELPERS
# ─────────────────────────────────────────────
COLOUR_NORMAL     = (0,   200,   0)    # green
COLOUR_FAST_MOVER = (0,   140, 255)    # orange
COLOUR_SUSPECT    = (0,     0, 200)    # red
COLOUR_VICTIM     = (200,   0, 200)    # magenta
COLOUR_LINE       = (0,   200, 200)    # cyan


def draw_frame(
    frame:   np.ndarray,
    objects: OrderedDict,
    bboxes:  dict[int, tuple],
    result:  dict,
    fps:     float,
) -> np.ndarray:
    """
    Render bounding boxes, connecting lines, and the dashboard overlay.
    """
    overlay = frame.copy()

    # Draw group-connection lines
    for group_key in result["groups"]:
        member_ids   = list(group_key)
        member_cents = [objects[mid] for mid in member_ids if mid in objects]
        for i in range(len(member_cents)):
            for j in range(i + 1, len(member_cents)):
                cv2.line(overlay,
                         tuple(member_cents[i]),
                         tuple(member_cents[j]),
                         COLOUR_LINE, 1)

    # Draw bounding boxes and labels
    for oid, centroid in objects.items():
        if oid in bboxes:
            (x, y, w, h) = bboxes[oid]
            if oid in result["victims"]:
                colour, label = COLOUR_VICTIM,     "VICTIM"
            elif oid in result["suspects"]:
                colour, label = COLOUR_SUSPECT,    "SUSPECT"
            elif oid in result["fast_movers"]:
                colour, label = COLOUR_FAST_MOVER, "FAST"
            else:
                colour, label = COLOUR_NORMAL,     f"P{oid}"

            cv2.rectangle(overlay, (x, y), (x + w, y + h), colour, 2)
            cv2.putText(overlay, label, (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

        cv2.circle(overlay, tuple(centroid), 4, (255, 255, 255), -1)

    # Alert banner
    if result["alert"]:
        h, w_frame = frame.shape[:2]
        banner_y   = h // 2 - 25
        cv2.rectangle(overlay, (0, banner_y), (w_frame, banner_y + 50),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, overlay)
        cv2.putText(overlay, "⚠  RAGGING ALERT DETECTED  ⚠",
                    (w_frame // 2 - 230, banner_y + 33),
                    cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
    else:
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, overlay)

    # Dashboard (top-left)
    dashboard_lines = [
        f"FPS: {fps:.1f}",
        f"Tracked: {len(objects)}",
        f"Flagged groups: {len(result['groups'])}",
        f"Suspects: {len(result['suspects'])}",
        f"Fast movers: {len(result['fast_movers'])}",
    ]
    if result["elapsed"]:
        max_elapsed = max(result["elapsed"].values())
        dashboard_lines.append(f"Max group time: {max_elapsed:.1f}s")

    for i, line in enumerate(dashboard_lines):
        cv2.putText(overlay, line, (10, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    return overlay


# ─────────────────────────────────────────────
# ALERT LOGGER
# ─────────────────────────────────────────────
class AlertLogger:
    def __init__(self, path: str = LOG_FILE):
        self.path = path
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame", "suspects", "victims",
                             "fast_movers", "group_count"])

    def log(self, frame_num: int, result: dict) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                frame_num,
                len(result["suspects"]),
                len(result["victims"]),
                len(result["fast_movers"]),
                len(result["groups"]),
            ])


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run(source: int | str = 0) -> None:
    """
    Main entry point.

    Args:
        source: Camera index (int) or RTSP / file path (str).
                Default 0 = first webcam.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    hog      = build_hog_detector()
    tracker  = CentroidTracker(max_disappeared=MAX_DISAPPEARED)
    analyser = BehaviourAnalyser()
    logger   = AlertLogger(LOG_FILE)

    frame_count  = 0
    fps_display  = 0.0
    fps_timer    = time.time()
    alert_logged = False   # debounce: log once per alert burst

    print("[INFO] Starting Anti-Ragging Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Stream ended.")
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optional: CLAHE for low-light robustness
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)

        # Detect persons
        rects = detect_persons(hog, gray)

        # Build id→bbox map for visualisation (centroid→bbox approximation)
        # Tracker returns id→centroid; we need to link back to bboxes
        # Strategy: after update, match each centroid to nearest input rect.
        objects    = tracker.update(rects)

        # Map object ID → closest bounding box
        bboxes: dict[int, tuple] = {}
        if rects and objects:
            input_cents = np.array(
                [(x + w // 2, y + h // 2) for (x, y, w, h) in rects]
            )
            for oid, cent in objects.items():
                dists = np.linalg.norm(input_cents - cent, axis=1)
                idx   = int(dists.argmin())
                bboxes[oid] = rects[idx]

        # Behavioural analysis
        result = analyser.analyse(objects, time.time())

        # Logging (debounced)
        if result["alert"]:
            if not alert_logged:
                logger.log(frame_count, result)
                alert_logged = True
        else:
            alert_logged = False

        # FPS calculation
        frame_count += 1
        if frame_count % 15 == 0:
            elapsed  = time.time() - fps_timer
            fps_display = 15 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        # Render
        output = draw_frame(frame, objects, bboxes, result, fps_display)
        cv2.imshow("Anti-Ragging Detection", output)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Alert log saved to '{LOG_FILE}'.")


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Rule-Based Anti-Ragging Detection System"
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Video source: camera index (0,1,...) or RTSP URL / file path"
    )
    args = parser.parse_args()

    # Allow numeric camera indices
    source = int(args.source) if args.source.isdigit() else args.source
    run(source)
