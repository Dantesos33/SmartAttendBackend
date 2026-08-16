import face_recognition
import cv2
import numpy as np
import os
import base64
import gc
from datetime import datetime
import json
import urllib.request
import threading

MIN_CONFIDENCE = 0.50
MAX_DETECT_EDGE = 2000
# Conservative quality gates for group photos. Small faces get a lower sharpness
# requirement so the fourth student is not discarded just because they are farther away.
MIN_FACE_SHARPNESS_LARGE = 16.0
MIN_FACE_SHARPNESS_SMALL = 10.0
FOCUS_GAP_RATIO = 1.35
FOCUS_RELATIVE_MIN = 0.42
YUNET_NORMAL_SCORE = 0.35
YUNET_MASK_SCORE = 0.20
MAX_ANNOTATED_EDGE = 1200
YUNET_MODEL_URL = "https://huggingface.co/pollen-robotics/face_detection_yunet_2023mar/resolve/main/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")


class ClassroomAttendanceSystem:
    def __init__(self, known_students_dir="known_students"):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []
        self.known_students_dir = known_students_dir
        self.metadata_path = os.path.join(known_students_dir, "metadata.json")
        self.sessions_path = "attendance_sessions.json"

        os.makedirs(self.known_students_dir, exist_ok=True)
        self.metadata = self._load_json(self.metadata_path, {})
        self.sessions = self._load_json(self.sessions_path, [])
        self._yunet_detector = None
        self._yunet_mask_detector = None
        self._yunet_lock = threading.Lock()
        self.load_known_students_from_dir()

    def _load_json(self, path, default):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json(self, path, data):
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as e:
            print(f"Error saving {path}: {e}")

    def _encoding_to_json(self, encoding):
        return json.dumps([float(x) for x in encoding])

    def _json_to_encoding(self, raw):
        if not raw:
            return None
        try:
            values = json.loads(raw)
            if not isinstance(values, list) or len(values) != 128:
                return None
            return np.array(values, dtype=np.float64)
        except Exception:
            return None

    def merge_student_encoding(self, student_id, name, encoding):
        if encoding is None:
            return
        if student_id in self.known_face_ids:
            idx = self.known_face_ids.index(student_id)
            self.known_face_encodings[idx] = encoding
            self.known_face_names[idx] = name
        else:
            self.known_face_encodings.append(encoding)
            self.known_face_ids.append(student_id)
            self.known_face_names.append(name)

    def load_db_encodings(self, rows):
        loaded = 0
        for student_id, name, encoding_json in rows:
            encoding = self._json_to_encoding(encoding_json)
            if encoding is None:
                continue
            self.merge_student_encoding(student_id, name, encoding)
            loaded += 1
        if loaded:
            print(f"Merged {loaded} face encoding(s) from database.")

    def encoding_for_student(self, student_id):
        if student_id not in self.known_face_ids:
            return None
        idx = self.known_face_ids.index(student_id)
        return self.known_face_encodings[idx]

    def load_known_students_from_dir(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []

        if not os.path.exists(self.known_students_dir):
            return

        for filename in os.listdir(self.known_students_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(self.known_students_dir, filename)
                stem = os.path.splitext(filename)[0]
                if not stem.isdigit():
                    continue
                student_id = int(stem)
                name = self.metadata.get(str(student_id), {}).get("name", f"Student {student_id}")
                self._register_encoding(filepath, student_id, name)

    def _register_encoding(self, image_path, student_id, name):
        try:
            image = face_recognition.load_image_file(image_path)
            image, _ = self._downscale_rgb(image, 1280)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="hog")
            if not locations:
                return False, "No face detected in image.", None
            if len(locations) > 1:
                return False, "Multiple faces detected — please use a photo with only one person.", None

            encodings = face_recognition.face_encodings(image, [locations[0]], num_jitters=1)
            if not encodings:
                return False, "Could not encode face from image.", None

            self.merge_student_encoding(student_id, name, encodings[0])
            return True, f"Successfully registered face for: {name}", encodings[0]
        except Exception as e:
            return False, f"Error processing image: {str(e)}", None

    def verify_face_quality(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            image, _ = self._downscale_rgb(image, 1280)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=1, model="hog")
            if len(locations) == 0:
                return False, "No face detected. Please upload a clear photo of your face."
            if len(locations) > 1:
                return False, "Multiple faces detected. Please upload a photo with only yourself in frame."
            return True, "Face detected clearly."
        except Exception as e:
            return False, f"Couldn't process this image: {str(e)}"

    def register_student_face(self, image_path, student_id, name, roll=None):
        safe_path = os.path.join(self.known_students_dir, f"{student_id}.jpg")
        if image_path != safe_path:
            image = face_recognition.load_image_file(image_path)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(safe_path, bgr)
        else:
            safe_path = image_path

        success, message, encoding = self._register_encoding(safe_path, student_id, name)
        if success:
            self.metadata[str(student_id)] = {"name": name, "roll": roll}
            self._save_json(self.metadata_path, self.metadata)
        elif os.path.exists(safe_path) and safe_path != image_path:
            os.remove(safe_path)
        return success, message, encoding

    def remove_student(self, student_id):
        if student_id not in self.known_face_ids:
            return False, f"No registered face for student {student_id}."
        try:
            index = self.known_face_ids.index(student_id)
            self.known_face_ids.pop(index)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)

            if str(student_id) in self.metadata:
                del self.metadata[str(student_id)]
                self._save_json(self.metadata_path, self.metadata)

            file_path = os.path.join(self.known_students_dir, f"{student_id}.jpg")
            if os.path.exists(file_path):
                os.remove(file_path)

            return True, f"Removed registered face for student {student_id}."
        except Exception as e:
            return False, f"Error removing student {student_id}: {str(e)}"

    def _encode_bgr_jpeg_base64(self, bgr_image, quality=75, max_edge=MAX_ANNOTATED_EDGE):
        try:
            h, w = bgr_image.shape[:2]
            if max(h, w) > max_edge:
                scale = max_edge / max(h, w)
                bgr_image = cv2.resize(
                    bgr_image,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buffer = cv2.imencode(".jpg", bgr_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        except Exception:
            return None

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=78, padded=False, max_edge=384):
        """Return a high-quality face crop without introducing avoidable JPEG pixelation.

        The crop is always taken from the original full-resolution attendance image.
        We never downscale a small face, because enlarging a tiny source cannot restore
        detail and was making profile photos look blocky. Larger crops are capped at
        1024px and receive only a very mild unsharp mask after resizing.
        """
        try:
            h, w = rgb_image.shape[:2]
            top, left = int(top), int(left)
            bottom, right = int(bottom), int(right)
            if padded:
                fh, fw = max(1, bottom - top), max(1, right - left)
                pad_y = int(fh * 0.45)
                pad_x = int(fw * 0.45)
                top -= pad_y; bottom += pad_y; left -= pad_x; right += pad_x
            top = max(0, top); left = max(0, left)
            bottom = min(h, bottom); right = min(w, right)
            if bottom <= top or right <= left:
                return None

            crop = np.ascontiguousarray(rgb_image[top:bottom, left:right])
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            longest = max(crop_bgr.shape[:2])
            if longest > max_edge:
                scale = float(max_edge) / longest
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (max(1, int(round(crop_bgr.shape[1] * scale))), max(1, int(round(crop_bgr.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            # Very mild sharpening preserves edge detail after JPEG encoding without
            # creating the harsh halos that made earlier crops look artificial.
            blurred = cv2.GaussianBlur(crop_bgr, (0, 0), 0.7)
            crop_bgr = cv2.addWeighted(crop_bgr, 1.08, blurred, -0.08, 0)
            return self._encode_bgr_jpeg_base64(crop_bgr, quality=quality, max_edge=max_edge)
        except Exception:
            return None

    def _get_yunet_detector(self, input_size, score_threshold=YUNET_NORMAL_SCORE):
        """Return a lightweight YuNet detector.

        The normal detector uses a 0.35 confidence threshold. A second, lower
        threshold is used only for masked/occluded-face recovery and is filtered
        by eye geometry before a box is accepted. This is important for the
        classroom photo: lowering YuNet globally creates false positives, while
        eye-supported recovery can find a face whose mouth/chin is covered.
        """
        if not hasattr(cv2, "FaceDetectorYN"):
            return None
        os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
        if not os.path.exists(YUNET_MODEL_PATH):
            tmp = YUNET_MODEL_PATH + ".download"
            try:
                with self._yunet_lock:
                    if not os.path.exists(YUNET_MODEL_PATH):
                        print("Downloading YuNet face detector model...")
                        urllib.request.urlretrieve(YUNET_MODEL_URL, tmp)
                        os.replace(tmp, YUNET_MODEL_PATH)
            except Exception as exc:
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
                print(f"YuNet model download failed: {exc}")
                return None
        try:
            # Keep one detector instance per threshold. Recreating ONNX sessions
            # for every pass is unnecessarily expensive on the API worker.
            if score_threshold <= YUNET_MASK_SCORE:
                detector = self._yunet_mask_detector
                if detector is None:
                    detector = cv2.FaceDetectorYN.create(
                        YUNET_MODEL_PATH, "", tuple(map(int, input_size)),
                        float(score_threshold), 0.30, 5000
                    )
                    self._yunet_mask_detector = detector
            else:
                detector = self._yunet_detector
                if detector is None:
                    detector = cv2.FaceDetectorYN.create(
                        YUNET_MODEL_PATH, "", tuple(map(int, input_size)),
                        float(score_threshold), 0.30, 5000
                    )
                    self._yunet_detector = detector
            return detector
        except Exception as exc:
            print(f"YuNet initialization failed: {exc}")
            return None

    @staticmethod
    def _yunet_row_to_location(row, image_shape, allow_mask_recovery=False):
        """Convert a YuNet row to a safe face box.

        For low-confidence masked recovery, require two eye landmarks inside the
        proposed box. This prevents the low YuNet threshold from turning necks,
        clothing folds, or background texture into faces.
        """
        h, w = image_shape[:2]
        values = [float(v) for v in row]
        if len(values) < 5:
            return None
        x, y, bw, bh, score = values[:5]
        if bw < 14 or bh < 14:
            return None
        if x < -bw or y < -bh or x > w or y > h:
            return None

        left = max(0, int(round(x)))
        top = max(0, int(round(y)))
        right = min(w, int(round(x + bw)))
        bottom = min(h, int(round(y + bh)))
        bw2, bh2 = right - left, bottom - top
        if bw2 < 14 or bh2 < 14:
            return None

        ratio = bw2 / float(max(1, bh2))
        if ratio < 0.48 or ratio > 1.65:
            return None

        if allow_mask_recovery:
            # YuNet stores right-eye and left-eye landmarks at indices 5..8.
            if len(values) < 9:
                return None
            re_x, re_y, le_x, le_y = values[5:9]
            eye_points = ((re_x, re_y), (le_x, le_y))
            inside = [
                left <= ex <= right and top <= ey <= bottom
                for ex, ey in eye_points
            ]
            if not all(inside):
                return None

            eye_distance = float(np.hypot(re_x - le_x, re_y - le_y))
            if eye_distance < max(4.0, bw2 * 0.13):
                return None
            if eye_distance > bw2 * 0.85:
                return None

            eye_y_gap = abs(re_y - le_y)
            if eye_y_gap > max(8.0, bh2 * 0.30):
                return None

            # Both eyes should occupy the upper ~60% of the box. This is a
            # useful discriminator against neck/shoulder detections.
            if max(re_y, le_y) > top + bh2 * 0.62:
                return None

        return (top, right, bottom, left)

    def _yunet_locations(self, bgr_image, mask_recovery=False):
        """Return YuNet face boxes with optional eye-supported mask recovery."""
        h, w = bgr_image.shape[:2]
        threshold = YUNET_MASK_SCORE if mask_recovery else YUNET_NORMAL_SCORE
        detector = self._get_yunet_detector((w, h), score_threshold=threshold)
        if detector is None:
            return []
        try:
            detector.setInputSize((w, h))
            _, detections = detector.detect(bgr_image)
            if detections is None:
                return []
            results = []
            for row in detections:
                score = float(row[4])
                if score < threshold:
                    continue
                loc = self._yunet_row_to_location(
                    row, bgr_image.shape, allow_mask_recovery=mask_recovery
                )
                if loc is not None:
                    results.append(loc)
            return results
        except Exception as exc:
            print(f"YuNet detection failed: {exc}")
            return []


    def classify_face_occlusion(self, rgb_image, location, encoding_available=False):
        """Stage-2 mask/occlusion classification only.

        Detection is intentionally NOT performed here.  The detector has a
        protected baseline and this method only decides whether an already
        detected face has strong evidence of lower-face covering.

        The classifier uses several independent cues and a small landmark
        upscale pass.  This is deliberately conservative: one weak cue must
        never turn a normal face into ``masked``.
        """
        top, right, bottom, left = [int(x) for x in location]
        h, w = rgb_image.shape[:2]
        top, left = max(0, top), max(0, left)
        bottom, right = min(h, bottom), min(w, right)
        if bottom <= top or right <= left:
            return "clear"

        crop = np.ascontiguousarray(rgb_image[top:bottom, left:right])
        ch, cw = crop.shape[:2]
        if ch < 36 or cw < 24:
            return "clear"

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            upper = gray[int(ch * 0.15):int(ch * 0.52), int(cw * 0.08):int(cw * 0.92)]
            lower = gray[int(ch * 0.48):int(ch * 0.90), int(cw * 0.08):int(cw * 0.92)]
            if upper.size == 0 or lower.size == 0:
                return "clear"

            upper_mean, lower_mean = float(np.mean(upper)), float(np.mean(lower))
            upper_std, lower_std = float(np.std(upper)), float(np.std(lower))
            ue = cv2.Canny(upper, 50, 130)
            le = cv2.Canny(lower, 50, 130)
            upper_edges = float(np.mean(ue > 0))
            lower_edges = float(np.mean(le > 0))

            # Landmarks are unreliable on tiny classroom faces at native size.
            # Upscale only this small crop; this does not affect Stage-1 detection.
            landmark_crop = crop
            if max(ch, cw) < 180:
                scale = min(4.0, 180.0 / max(ch, cw))
                landmark_crop = cv2.resize(
                    crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )

            landmarks = {}
            try:
                lm = face_recognition.face_landmarks(landmark_crop, model="large")
                if lm:
                    landmarks = lm[0]
            except Exception:
                landmarks = {}

            eyes = bool(landmarks.get("left_eye")) and bool(landmarks.get("right_eye"))
            mouth = bool(landmarks.get("top_lip")) or bool(landmarks.get("bottom_lip"))
            nose = bool(landmarks.get("nose_bridge")) or bool(landmarks.get("nose_tip"))

            # Score independent signals instead of relying on one brittle rule.
            score = 0

            # Eyes visible but mouth structure missing is a strong occlusion cue.
            if eyes and not mouth:
                score += 2

            # If eyes are visible but both mouth and nose structure are absent,
            # this is particularly useful for niqab/face-covering cases.
            if eyes and not mouth and not nose:
                score += 2

            # Covered lower faces are commonly flatter/less textured than the
            # upper face.  Require a meaningful relative difference, not a fixed
            # brightness threshold so lighting/skin tone do not dominate.
            texture_flat = (
                lower_std < max(upper_std * 0.78, 9.0) and
                lower_edges < max(upper_edges * 0.72, 0.016)
            )
            if texture_flat:
                score += 1

            # A dark/opaque lower covering is an additional cue, but it is never
            # sufficient by itself because shadows and hair can look similar.
            dark_cover = (
                upper_mean > 45 and
                lower_mean < upper_mean * 0.72 and
                lower_std < max(upper_std * 0.82, 9.0)
            )
            if dark_cover:
                score += 1

            # Very low lower-face detail combined with visible eyes is useful for
            # masks/niqabs even when the landmark detector only finds the eyes.
            if eyes and lower_edges < max(upper_edges * 0.82, 0.018):
                score += 1

            # Require strong evidence.  This reduces false positives on sideways,
            # downward-facing and low-resolution clear faces.
            if score >= 3:
                return "masked"
            return "clear"
        except Exception:
            return "clear"

    @staticmethod
    def _fallback_occlusion_classification(rgb_crop):
        """Safe fallback when Haar cascades are unavailable.

        This intentionally returns clear unless there is a strong lower-face
        occlusion signal. It is better to keep a detected face in the pipeline
        than to crash or silently discard it when an optional OpenCV component
        is missing.
        """
        try:
            h, w = rgb_crop.shape[:2]
            if h < 24 or w < 24:
                return "clear"
            gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
            upper = gray[int(h * 0.18):int(h * 0.52), :]
            lower = gray[int(h * 0.52):int(h * 0.92), :]
            if upper.size == 0 or lower.size == 0:
                return "clear"
            # A very low-variance, strongly darker lower-face region is a useful
            # conservative signal for masks/niqab, without rejecting normal
            # faces based on colour.
            upper_mean = float(np.mean(upper))
            lower_mean = float(np.mean(lower))
            upper_std = float(np.std(upper))
            lower_std = float(np.std(lower))
            if upper_mean > 35 and lower_mean < upper_mean * 0.58 and lower_std < upper_std * 0.9:
                return "masked"
            return "clear"
        except Exception:
            return "clear"

    def _downscale_rgb(self, rgb_image, max_edge):
        h, w = rgb_image.shape[:2]
        longest = max(h, w)
        if longest <= max_edge:
            return np.ascontiguousarray(rgb_image), 1.0
        scale = max_edge / float(longest)
        resized = cv2.resize(
            rgb_image,
            (int(round(w * scale)), int(round(h * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
        return np.ascontiguousarray(resized), scale

    def prepare_known_faces(self, db_encoding_rows=None, allowed_student_ids=None):
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []

        if db_encoding_rows:
            self.load_db_encodings(db_encoding_rows)

        # Recognition identity is one student, not one database encoding row.
        # Collapse duplicate rows for the same student so a student can never
        # appear multiple times in the candidate pool.
        seen = set()
        dedup_encodings, dedup_ids, dedup_names = [], [], []
        for enc, sid, name in zip(self.known_face_encodings, self.known_face_ids, self.known_face_names):
            sid = int(sid)
            if sid in seen:
                continue
            seen.add(sid)
            dedup_encodings.append(enc)
            dedup_ids.append(sid)
            dedup_names.append(name)
        self.known_face_encodings = dedup_encodings
        self.known_face_ids = dedup_ids
        self.known_face_names = dedup_names

        target_ids = allowed_student_ids
        if target_ids is None:
            target_ids = [
                int(stem)
                for stem in (
                    os.path.splitext(name)[0]
                    for name in os.listdir(self.known_students_dir)
                    if name.lower().endswith((".png", ".jpg", ".jpeg"))
                    and os.path.splitext(name)[0].isdigit()
                )
            ]

        for student_id in target_ids:
            if student_id in self.known_face_ids:
                continue
            filepath = os.path.join(self.known_students_dir, f"{student_id}.jpg")
            if not os.path.exists(filepath):
                continue
            name = self.metadata.get(str(student_id), {}).get("name", f"Student {student_id}")
            self._register_encoding(filepath, student_id, name)

    @staticmethod
    def _box_iou(a, b):
        at, ar, ab, al = a
        bt, br, bb, bl = b
        left = max(al, bl)
        top = max(at, bt)
        right = min(ar, br)
        bottom = min(ab, bb)
        if right <= left or bottom <= top:
            return 0.0
        inter = float((right - left) * (bottom - top))
        area_a = float(max(1, ar - al) * max(1, ab - at))
        area_b = float(max(1, br - bl) * max(1, bb - bt))
        return inter / max(1.0, area_a + area_b - inter)

    @staticmethod
    def _focus_score(rgb_crop):
        """Return a scale/contrast-normalized facial-detail score.

        This intentionally avoids ranking all faces against one another. In a
        classroom photo, a focused small face can legitimately have a lower raw
        Laplacian variance than a larger blurred face. We therefore measure
        fine detail relative to local contrast and edge structure.
        """
        if rgb_crop is None or rgb_crop.size == 0:
            return 0.0
        try:
            h, w = rgb_crop.shape[:2]
            if min(h, w) < 12:
                return 0.0

            # The upper/central face area contains eyes and facial detail and is
            # much less contaminated by shirts, shoulders and background edges.
            y0, y1 = int(h * 0.08), int(h * 0.82)
            x0, x1 = int(w * 0.12), int(w * 0.88)
            crop = rgb_crop[y0:y1, x0:x1]
            if crop.size == 0:
                return 0.0

            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_CUBIC)
            gray_f = gray.astype(np.float32)

            # Blur removes high-frequency facial detail.  Measuring the detail
            # left after a tiny Gaussian blur is substantially less sensitive to
            # face size than raw Laplacian variance.
            smooth = cv2.GaussianBlur(gray_f, (5, 5), 0)
            detail = float(np.std(gray_f - smooth))
            contrast = float(np.std(gray_f)) + 1.0
            detail_ratio = detail / contrast

            gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
            gradient = float(np.mean(np.sqrt(gx * gx + gy * gy)))
            gradient_norm = gradient / (contrast + 1.0)

            edges = cv2.Canny(gray, 50, 120)
            edge_density = float(np.mean(edges > 0))

            # Dimensionless score. Higher = more facial detail/focus.
            return (
                0.55 * detail_ratio
                + 0.30 * min(1.0, gradient_norm / 2.5)
                + 0.15 * min(1.0, edge_density / 0.10)
            )
        except Exception:
            return 0.0

    def _filter_face_locations_by_focus(self, rgb_image, locations):
        """Remove only *clearly* out-of-focus faces.

        The previous relative-gap algorithm always tried to split a group into
        "sharp" and "soft" faces. That was wrong for this application because
        the fourth, genuinely focused student could be the lower-scoring face.
        This version uses a conservative absolute blur gate and requires more
        than one weak-detail signal before discarding a face.
        """
        h, w = rgb_image.shape[:2]
        scored = []
        for loc in locations:
            t, r, b, l = [int(v) for v in loc]
            t = max(0, t); l = max(0, l); b = min(h, b); r = min(w, r)
            if b <= t or r <= l:
                continue
            crop = rgb_image[t:b, l:r]
            score = self._focus_score(crop)
            scored.append(((t, r, b, l), score, min(r - l, b - t)))

        if not scored:
            return []

        kept = []
        filtered = []
        for loc, score, min_dim in scored:
            # Do not filter tiny faces solely on focus. At that size there is
            # not enough pixel information for a reliable blur decision.
            if min_dim < 32:
                kept.append(loc)
                continue

            # Strict enough that ordinary JPEG/compression softness survives,
            # but a deliberately blurred/background face is rejected.
            detail_floor = 0.055 if min_dim >= 55 else 0.045
            if score < detail_floor:
                filtered.append((loc, score, min_dim))
            else:
                kept.append(loc)

        print(
            "Focus scores -> "
            + ", ".join(
                f"{score:.3f}@{dim}px" for _, score, dim in scored
            )
        )
        if filtered:
            print(
                "Focus filtered -> "
                + ", ".join(f"{score:.3f}@{dim}px" for _, score, dim in filtered)
            )
        return kept

    def _dedupe_face_locations(self, locations, image_shape):
        """Merge detections from multiple detectors without losing small faces."""
        h, w = image_shape[:2]
        candidates = []
        for loc in locations:
            t, r, b, l = [int(v) for v in loc]
            t = max(0, min(h - 1, t)); b = max(0, min(h, b))
            l = max(0, min(w - 1, l)); r = max(0, min(w, r))
            bw, bh = r - l, b - t
            if bw < 14 or bh < 14:
                continue
            ratio = bw / float(max(1, bh))
            if ratio < 0.38 or ratio > 2.2:
                continue
            candidates.append((t, r, b, l))

        # Prefer larger, more complete detections when two detectors found the
        # same person. This keeps the small-face recovery boxes only when they
        # don't overlap an already good detection.
        candidates.sort(key=lambda x: (x[2]-x[0])*(x[1]-x[3]), reverse=True)
        kept = []
        for candidate in candidates:
            if any(self._box_iou(candidate, existing) >= 0.35 for existing in kept):
                continue
            kept.append(candidate)

        kept.sort(key=lambda x: (x[0], x[3]))
        return kept

    @staticmethod
    def _map_rotated_location(location, original_shape, rotation):
        """Map a box from a cv2.rotate image back to the original image."""
        h, w = original_shape[:2]
        t, r, b, l = [int(v) for v in location]
        if rotation == "cw":
            return (l, h - t, r, h - b)
        if rotation == "ccw":
            return (w - r, b, w - l, t)
        if rotation == "180":
            return (h - b, w - l, h - t, w - r)
        return (t, r, b, l)

    def _eye_pair_face_proposals(self, rgb_image):
        """Recover masked faces when the full-face detector misses the mouth/chin.

        This is deliberately conservative. Two eyes must be detected on roughly
        the same horizontal line and the resulting face box must have sensible
        proportions. Coordinates are built in the original image coordinate
        system, so a proposal can never accidentally land at (0, 0).
        """
        try:
            if not hasattr(cv2, "CascadeClassifier"):
                return []
            cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
            if not os.path.exists(cascade_path):
                return []
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                return []

            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            # Keep this pass modest; it is only a recovery mechanism.
            gray_small, scale = self._downscale_rgb(
                cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), 1600
            )
            gray_small = cv2.cvtColor(gray_small, cv2.COLOR_RGB2GRAY)
            eyes = cascade.detectMultiScale(
                gray_small,
                scaleFactor=1.08,
                minNeighbors=5,
                minSize=(10, 10),
            )
            if len(eyes) < 2:
                return []

            inv = 1.0 / scale
            centers = []
            for ex, ey, ew, eh in eyes:
                centers.append((
                    (ex + ew * 0.5) * inv,
                    (ey + eh * 0.5) * inv,
                    max(ew, eh) * inv,
                ))

            proposals = []
            used = set()
            for i, (cx1, cy1, s1) in enumerate(centers):
                best = None
                best_score = float("inf")
                for j, (cx2, cy2, s2) in enumerate(centers):
                    if i == j or j in used:
                        continue
                    dy = abs(cy1 - cy2)
                    dx = abs(cx1 - cx2)
                    if dx < max(s1, s2) * 1.5:
                        continue
                    if dy > max(s1, s2) * 1.0:
                        continue
                    size_ratio = max(s1, s2) / max(1.0, min(s1, s2))
                    if size_ratio > 1.8:
                        continue
                    score = dy / max(1.0, max(s1, s2)) + abs(size_ratio - 1.0)
                    if score < best_score:
                        best_score = score
                        best = j

                if best is None:
                    continue

                cx2, cy2, s2 = centers[best]
                eye_dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
                if eye_dist < 12 or eye_dist > 220:
                    continue

                # Build a conventional face rectangle around the eyes.
                left = int(round(min(cx1, cx2) - eye_dist * 0.62))
                right = int(round(max(cx1, cx2) + eye_dist * 0.62))
                top = int(round(min(cy1, cy2) - eye_dist * 0.62))
                bottom = int(round(min(cy1, cy2) + eye_dist * 1.45))
                h, w = rgb_image.shape[:2]
                left = max(0, left); right = min(w, right)
                top = max(0, top); bottom = min(h, bottom)
                bw, bh = right - left, bottom - top
                if bw < 20 or bh < 20:
                    continue
                ratio = bw / float(max(1, bh))
                if 0.48 <= ratio <= 1.65:
                    proposals.append((top, right, bottom, left))
                used.add(best)
            return proposals
        except Exception as exc:
            print(f"Eye-pair recovery failed: {exc}")
            return []

    def _detect_face_locations(self, rgb_image):
        """Detect faces independently of recognition, including difficult group-photo faces.

        The detector intentionally combines several lightweight passes:
        - frontal HOG for normal faces;
        - OpenCV frontal/profile cascades for small/partial faces;
        - 90/270-degree HOG recovery for faces looking down/up or presented at
          difficult orientations;
        - eye-pair proposals for masks/niqabs where only the eyes are visible.

        Every pass is sequential and downscaled so Railway memory stays bounded.
        """
        original = np.ascontiguousarray(rgb_image)
        h, w = original.shape[:2]
        detect_img, scale = self._downscale_rgb(original, MAX_DETECT_EDGE)
        inv_scale = 1.0 / scale
        raw = []

        def restore(loc):
            t, r, b, l = loc
            return (
                max(0, int(round(t * inv_scale))),
                min(w, int(round(r * inv_scale))),
                min(h, int(round(b * inv_scale))),
                max(0, int(round(l * inv_scale))),
            )

        # 1) Primary frontal HOG.
        try:
            for loc in face_recognition.face_locations(
                detect_img, number_of_times_to_upsample=1, model="hog"
            ):
                raw.append(restore(loc))
        except Exception as exc:
            print(f"Primary HOG detection failed: {exc}")

        # 2) YuNet is the primary recovery detector. The normal pass is kept
        # conservative so it does not create background/neck false positives.
        try:
            yunet_input = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
            for loc in self._yunet_locations(yunet_input, mask_recovery=False):
                raw.append(restore(loc))

            # 2b) Low-confidence YuNet recovery is used ONLY with eye geometry.
            # This is specifically for masked/niqab faces where the lower face
            # is hidden and the normal YuNet confidence can be too low.
            for loc in self._yunet_locations(yunet_input, mask_recovery=True):
                raw.append(restore(loc))
            del yunet_input
        except Exception as exc:
            print(f"YuNet recovery failed: {exc}")

        # 2c) Eye-pair fallback for covered faces. It is conservative and uses
        # original-image coordinates, preventing top-left drift.
        try:
            for loc in self._eye_pair_face_proposals(detect_img):
                raw.append(restore(loc))
        except Exception as exc:
            print(f"Eye-pair fallback failed: {exc}")

        # 3) Small-face tile recovery. A group photo can contain people whose
        # faces are only a few dozen pixels wide. HOG on the whole image can
        # miss those, while a sequential overlapping tile gives the detector
        # much more face resolution without the memory cost of full-image
        # upsample=2/3.
        th, tw = detect_img.shape[:2]
        tile_w = max(420, int(tw * 0.62))
        tile_h = max(360, int(th * 0.62))
        step_x = max(1, int(tile_w * 0.58))
        step_y = max(1, int(tile_h * 0.58))
        xs = sorted(set([0, max(0, tw - tile_w)] + list(range(0, max(1, tw - tile_w + 1), step_x))))
        ys = sorted(set([0, max(0, th - tile_h)] + list(range(0, max(1, th - tile_h + 1), step_y))))
        for y0 in ys:
            for x0 in xs:
                tile = detect_img[y0:min(th, y0+tile_h), x0:min(tw, x0+tile_w)]
                if tile.shape[0] < 300 or tile.shape[1] < 300:
                    continue
                try:
                    # Upsample only the tile, not the complete classroom image.
                    for loc in face_recognition.face_locations(
                        tile, number_of_times_to_upsample=2, model="hog"
                    ):
                        t, r, b, l = loc
                        raw.append(restore((t+y0, r+x0, b+y0, l+x0)))
                except Exception as exc:
                    print(f"HOG tile detection failed at ({x0},{y0}): {exc}")
                del tile
                gc.collect()

        # 4) Orientation recovery.
        for rotation, rotated in (
            ("cw", cv2.rotate(detect_img, cv2.ROTATE_90_CLOCKWISE)),
            ("ccw", cv2.rotate(detect_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        ):
            try:
                for loc in face_recognition.face_locations(
                    rotated, number_of_times_to_upsample=1, model="hog"
                ):
                    mapped = self._map_rotated_location(loc, detect_img.shape, rotation)
                    raw.append(restore(mapped))
            except Exception as exc:
                print(f"Rotated HOG ({rotation}) failed: {exc}")
            del rotated
            gc.collect()

        # 5) A second YuNet pass on a 1.35x image catches distant/partially
        # occluded faces that are below the effective size of the first pass.
        try:
            enlarged = cv2.resize(detect_img, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            enlarged_bgr = cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)
            for loc in self._yunet_locations(enlarged_bgr, mask_recovery=False):
                t,r,b,l = loc
                raw.append(restore((int(t/1.35), int(r/1.35), int(b/1.35), int(l/1.35))))
            for loc in self._yunet_locations(enlarged_bgr, mask_recovery=True):
                t,r,b,l = loc
                raw.append(restore((int(t/1.35), int(r/1.35), int(b/1.35), int(l/1.35))))
            del enlarged_bgr
            del enlarged
            gc.collect()
        except Exception as exc:
            print(f"Enlarged YuNet recovery failed: {exc}")

        final_locations = self._dedupe_face_locations(raw, original.shape)
        before_focus = len(final_locations)
        final_locations = self._filter_face_locations_by_focus(original, final_locations)
        print(
            f"Detected {len(final_locations)} face(s) in image "
            f"(deduped={before_focus}, focus_filtered={before_focus-len(final_locations)})."
        )
        return final_locations

    def _assign_faces_to_students(
        self,
        face_encodings_by_index: dict[int, np.ndarray | None],
        tolerance: float,
        allowed_set: set | None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict[int, tuple[int | None, str, float]]:
        """Assign clear detected faces to enrolled students without duplicates.

        Recognition is intentionally independent from Stage-1 detection.  This
        matcher is tuned for classroom photos where an enrolled student's face
        can be farther from the profile-photo embedding because of distance,
        pose, lighting, or expression.  We therefore use a two-tier acceptance
        rule instead of a single overly-strict threshold:

        * strong matches are accepted directly;
        * weaker matches are accepted only when they have a clear margin over
          the next-best student for that face.

        Assignment is global and one-to-one so the same student cannot be marked
        present for multiple detected faces.
        """
        if not face_encodings_by_index or not self.known_face_encodings:
            return {i: (None, "Unknown", 0.0) for i in face_encodings_by_index}

        # Build one best distance per (face, student).  The database loader has
        # already collapsed duplicate rows for a student.
        candidates_by_face: dict[int, list[tuple[float, int, str, float]]] = {}
        for face_index, encoding in face_encodings_by_index.items():
            if encoding is None:
                candidates_by_face[face_index] = []
                continue

            try:
                distances = face_recognition.face_distance(
                    self.known_face_encodings, encoding
                )
            except Exception as exc:
                print(f"Face-distance calculation failed for face {face_index}: {exc}")
                candidates_by_face[face_index] = []
                continue

            options = []
            for idx, raw_distance in enumerate(distances):
                sid = int(self.known_face_ids[idx])
                if allowed_set is not None and sid not in allowed_set:
                    continue
                distance = float(raw_distance)
                confidence = max(0.0, 1.0 - distance)

                # Hard ceiling.  Never allow a genuinely distant embedding to
                # become a classroom match merely because the student is still
                # unused.
                if distance > max(float(tolerance), 0.64):
                    continue
                options.append(
                    (distance, sid, self.known_face_names[idx], confidence)
                )

            options.sort(key=lambda item: item[0])
            candidates_by_face[face_index] = options

        # Convert each face's candidate list into an acceptance candidate.  A
        # relaxed match is allowed for difficult classroom faces only when the
        # best student is clearly separated from the second-best student.
        candidates = []
        for face_index, options in candidates_by_face.items():
            if not options:
                continue

            best = options[0]
            best_distance, sid, name, confidence = best
            second_distance = options[1][0] if len(options) > 1 else float("inf")
            margin = second_distance - best_distance

            strong_limit = min(float(tolerance), 0.58)
            relaxed_limit = max(float(tolerance), 0.62)
            required_confidence = max(0.50, float(min_confidence))

            accepted = False
            # Attendance policy: anything below 50% is always Unrecognized.
            if best_distance <= strong_limit and confidence >= required_confidence:
                accepted = True
            elif (
                best_distance <= relaxed_limit
                and confidence >= required_confidence
                and margin >= 0.075
            ):
                accepted = True

            if accepted:
                candidates.append(
                    (best_distance, face_index, sid, name, confidence)
                )

        # Global one-to-one assignment.  Stronger matches are always considered
        # before relaxed matches.  This prevents two classroom faces from being
        # given the same enrolled student's name.
        candidates.sort(key=lambda item: (item[0], item[1]))
        assignments: dict[int, tuple[int | None, str, float]] = {}
        assigned_faces = set()
        assigned_students = set()

        for distance, face_index, sid, name, confidence in candidates:
            if face_index in assigned_faces or sid in assigned_students:
                continue
            assignments[face_index] = (sid, name, confidence)
            assigned_faces.add(face_index)
            assigned_students.add(sid)

        for face_index in face_encodings_by_index:
            assignments.setdefault(face_index, (None, "Unknown", 0.0))
        return assignments

    def recognize_classroom(
        self,
        classroom_image_path,
        tolerance=0.50,
        allowed_student_ids=None,
        db_encoding_rows=None,
    ):
        self.prepare_known_faces(
            db_encoding_rows=db_encoding_rows,
            allowed_student_ids=allowed_student_ids,
        )

        has_registered_faces = bool(self.known_face_encodings)
        allowed_set = set(allowed_student_ids) if allowed_student_ids else None

        enrolled_with_faces = []
        if allowed_set is not None:
            enrolled_with_faces = [
                sid for sid in allowed_set if sid in self.known_face_ids
            ]

        classroom_image = face_recognition.load_image_file(classroom_image_path)
        classroom_image = np.ascontiguousarray(classroom_image)
        classroom_image_cv = cv2.cvtColor(classroom_image, cv2.COLOR_RGB2BGR)

        face_locations = self._detect_face_locations(classroom_image)

        # Batch encode all detected faces safely
        face_encodings_by_index = {}
        if face_locations:
            try:
                encodings = face_recognition.face_encodings(
                    classroom_image, 
                    known_face_locations=face_locations, 
                    num_jitters=1
                )
                for idx, enc in enumerate(encodings):
                    face_encodings_by_index[idx] = enc
            except Exception as e:
                print(f"Face encoding error: {e}")
                for idx in range(len(face_locations)):
                    face_encodings_by_index[idx] = None

        assignments = {}
        if has_registered_faces:
            assignments = self._assign_faces_to_students(
                face_encodings_by_index,
                tolerance=tolerance,
                allowed_set=allowed_set,
            )

        present_student_ids = []
        unknown_faces = 0
        face_details = []

        for face_index, location in enumerate(face_locations):
            top, right, bottom, left = location
            face_encoding = face_encodings_by_index.get(face_index)

            student_id = None
            name = "Unknown"
            confidence = 0.0

            if face_encoding is None or not has_registered_faces:
                unknown_faces += 1
            else:
                student_id, name, confidence = assignments.get(
                    face_index, (None, "Unknown", 0.0)
                )
                if student_id is None:
                    unknown_faces += 1
                elif student_id not in present_student_ids:
                    present_student_ids.append(student_id)

            face_details.append({
                "face_index": face_index,
                "student_id": student_id,
                "name": name,
                "confidence": float(confidence),
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "crop_base64": self._encode_rgb_crop_base64(
                    classroom_image, top, right, bottom, left
                ),
            })

            # Draw green for recognized, red for unknown
            color = (0, 255, 0) if student_id is not None else (0, 0, 255)
            cv2.rectangle(classroom_image_cv, (left, top), (right, bottom), color, 3)

            label = f"{name} ({confidence:.0%})" if student_id is not None else "Unknown"
            label_y = max(top - 10, 20)
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(
                classroom_image_cv,
                (left, label_y - h_txt - 4),
                (left + w_txt + 6, label_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                classroom_image_cv,
                label,
                (left + 3, label_y - 2),
                cv2.FONT_HERSHEY_DUPLEX,
                0.55,
                (255, 255, 255),
                1,
            )

        if allowed_set is not None:
            absent_student_ids = [
                sid for sid in allowed_set if sid not in present_student_ids
            ]
            registered_in_section = enrolled_with_faces or []
        else:
            absent_student_ids = [
                sid for sid in self.known_face_ids if sid not in present_student_ids
            ]
            registered_in_section = list(self.known_face_ids)

        attendance_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "total_registered": len(registered_in_section),
            "present_student_ids": present_student_ids,
            "absent_student_ids": absent_student_ids,
            "present_count": len(present_student_ids),
            "absent_count": len(absent_student_ids),
            "unknown_faces": unknown_faces,
            "faces_detected": len(face_locations),
            "enrolled_with_face_photos": len(registered_in_section),
            "recognition_available": has_registered_faces,
            "warning_message": (
                None
                if has_registered_faces
                else "No enrolled students have profile photos registered on the server."
            ),
            "face_details": face_details,
        }

        os.makedirs("output", exist_ok=True)
        annotated_path = f"output/annotated_classroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(annotated_path, classroom_image_cv)
        attendance_data["annotated_image_path"] = annotated_path
        attendance_data["annotated_image_base64"] = self._encode_bgr_jpeg_base64(
            classroom_image_cv, quality=72, max_edge=MAX_ANNOTATED_EDGE
        )

        del classroom_image
        gc.collect()

        result_message = (
            "Recognition complete!"
            if has_registered_faces
            else "Face detection complete — no registered student photos to match against."
        )
        return attendance_data, result_message