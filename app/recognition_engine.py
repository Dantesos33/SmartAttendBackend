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

MIN_CONFIDENCE = 0.45
MAX_DETECT_EDGE = 2000
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

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=98, padded=False):
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
            if longest > 1024:
                scale = 1024.0 / longest
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (max(1, int(round(crop_bgr.shape[1] * scale))), max(1, int(round(crop_bgr.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            # Very mild sharpening preserves edge detail after JPEG encoding without
            # creating the harsh halos that made earlier crops look artificial.
            blurred = cv2.GaussianBlur(crop_bgr, (0, 0), 0.7)
            crop_bgr = cv2.addWeighted(crop_bgr, 1.08, blurred, -0.08, 0)
            return self._encode_bgr_jpeg_base64(crop_bgr, quality=98, max_edge=1024)
        except Exception:
            return None

    def _get_yunet_detector(self, input_size):
        """Return a lightweight YuNet detector. The model is downloaded once if absent.

        YuNet is used instead of Haar Cascade because some Railway OpenCV builds do
        not expose CascadeClassifier. It is also substantially better for small,
        partially occluded and non-frontal faces.
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
            detector = cv2.FaceDetectorYN.create(
                YUNET_MODEL_PATH, "", tuple(map(int, input_size)),
                0.35, 0.30, 5000
            )
            return detector
        except Exception as exc:
            print(f"YuNet initialization failed: {exc}")
            return None

    def _yunet_locations(self, bgr_image):
        h, w = bgr_image.shape[:2]
        detector = self._get_yunet_detector((w, h))
        if detector is None:
            return []
        try:
            detector.setInputSize((w, h))
            _, detections = detector.detect(bgr_image)
            if detections is None:
                return []
            results = []
            for row in detections:
                x, y, bw, bh, score = [float(v) for v in row[:5]]
                if score < 0.35 or bw < 10 or bh < 10:
                    continue
                results.append((int(y), int(x+bw), int(y+bh), int(x)))
            return results
        except Exception as exc:
            print(f"YuNet detection failed: {exc}")
            return []


    def classify_face_occlusion(self, rgb_image, location, encoding_available=False):
        """Conservative Stage-2 mask/occlusion classifier.

        IMPORTANT: this function does not perform detection. Stage-1 locations are
        passed in unchanged. Recognition availability must also not automatically
        make a face "clear": masked students can still produce an embedding.

        A face is marked masked only when there is actual lower-face occlusion
        evidence. Ordinary profile/downward/low-resolution faces remain clear and
        can continue to enrollment matching.
        """
        top, right, bottom, left = [int(x) for x in location]
        h, w = rgb_image.shape[:2]
        top = max(0, top); left = max(0, left)
        bottom = min(h, bottom); right = min(w, right)
        if bottom <= top or right <= left:
            return "clear"

        crop = np.ascontiguousarray(rgb_image[top:bottom, left:right])
        ch, cw = crop.shape[:2]
        if ch < 48 or cw < 32:
            # Tiny faces are intentionally never classified as masked from weak
            # image statistics. They remain clear/unrecognized.
            return "clear"

        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            # Focus on the facial area; ignore the very top hair/background and
            # the bottom shirt/background region.
            upper = gray[int(ch * 0.16):int(ch * 0.50), int(cw * 0.10):int(cw * 0.90)]
            lower = gray[int(ch * 0.50):int(ch * 0.88), int(cw * 0.10):int(cw * 0.90)]
            if upper.size == 0 or lower.size == 0:
                return "clear"

            upper_mean = float(np.mean(upper))
            lower_mean = float(np.mean(lower))
            upper_std = float(np.std(upper))
            lower_std = float(np.std(lower))

            # Edge density is useful because fabric masks/niqabs tend to make the
            # lower face relatively uniform while a visible mouth/chin has more
            # local structure. This is only a supporting signal.
            upper_edges = cv2.Canny(upper, 60, 140)
            lower_edges = cv2.Canny(lower, 60, 140)
            upper_edge_density = float(np.mean(upper_edges > 0))
            lower_edge_density = float(np.mean(lower_edges > 0))

            landmarks = {}
            try:
                lm = face_recognition.face_landmarks(crop, model="large")
                if lm:
                    landmarks = lm[0]
            except Exception:
                landmarks = {}

            has_both_eyes = bool(landmarks.get("left_eye")) and bool(landmarks.get("right_eye"))
            has_mouth = bool(landmarks.get("top_lip")) or bool(landmarks.get("bottom_lip"))
            has_nose = bool(landmarks.get("nose_bridge")) or bool(landmarks.get("nose_tip"))

            # Strong mask/niqab signal:
            # - eyes are visible, while mouth is absent, AND
            # - lower face is unusually uniform OR substantially less detailed.
            eye_only_occlusion = (
                has_both_eyes and
                not has_mouth and
                (lower_std < max(upper_std * 0.78, 10.0) or
                 lower_edge_density < max(upper_edge_density * 0.58, 0.018))
            )

            # Strong fabric/occlusion signal even if landmarks are incomplete.
            # Requiring both intensity and texture differences avoids classifying
            # ordinary shadows, sideways faces, or low-resolution faces as masks.
            dark_flat_lower = (
                upper_mean > 45 and
                lower_mean < upper_mean * 0.68 and
                lower_std < max(upper_std * 0.72, 10.0) and
                lower_edge_density < max(upper_edge_density * 0.62, 0.020)
            )

            # A niqab can cover most of the lower face without being particularly
            # dark. In that case the reliable signal is visible eyes + missing
            # mouth/nose structure + low lower-face detail.
            niqab_like = (
                has_both_eyes and
                not has_mouth and
                not has_nose and
                lower_std < max(upper_std * 0.88, 12.0) and
                lower_edge_density < max(upper_edge_density * 0.72, 0.022)
            )

            if eye_only_occlusion or dark_flat_lower or niqab_like:
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

    @staticmethod
    def _downscale_rgb(rgb_image, max_edge):
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
        # Legacy compatibility hook. Haar cascades are intentionally not used
        # because some production OpenCV builds omit CascadeClassifier.
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

        # 2) YuNet is the primary recovery detector. It handles small,
        # partially occluded, profile and masked faces much better than Haar.
        try:
            yunet_input = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
            for loc in self._yunet_locations(yunet_input):
                raw.append(restore(loc))
            del yunet_input
        except Exception as exc:
            print(f"YuNet recovery failed: {exc}")

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
            for loc in self._yunet_locations(cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)):
                t,r,b,l = loc
                raw.append(restore((int(t/1.35), int(r/1.35), int(b/1.35), int(l/1.35))))
            del enlarged
            gc.collect()
        except Exception as exc:
            print(f"Enlarged YuNet recovery failed: {exc}")

        final_locations = self._dedupe_face_locations(raw, original.shape)
        print(f"Detected {len(final_locations)} face(s) in image.")
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

            accepted = False
            if best_distance <= strong_limit and confidence >= 0.40:
                accepted = True
            elif (
                best_distance <= relaxed_limit
                and confidence >= max(0.36, float(min_confidence))
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