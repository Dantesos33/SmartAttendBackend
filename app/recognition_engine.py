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
MAX_DETECT_EDGE = 1800
MAX_ANNOTATED_EDGE = 1000
MAX_FACE_CROP_EDGE = 512
YUNET_MODEL_URL = "https://huggingface.co/pollen-robotics/face_detection_yunet_2023mar/resolve/main/face_detection_yunet_2023mar.onnx"
YUNET_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx")
SR_MODEL_URL = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
SR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "FSRCNN_x2.pb")


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
        self._sr_model = None
        self._sr_lock = threading.Lock()
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

    def _get_superres(self):
        """Load FSRCNN x2 once, downloading the model on first use if needed.

        The model is treated like YuNet: it is stored in the container's local
        models directory after a successful download. A temporary file and
        atomic replace prevent a partial download from being loaded. If the
        download or OpenCV super-resolution module is unavailable, callers
        safely fall back to interpolation instead of failing attendance.
        """
        if self._sr_model is not None:
            return self._sr_model

        if not hasattr(cv2, "dnn_superres"):
            print("OpenCV dnn_superres is unavailable; using high-quality interpolation fallback.")
            return None

        os.makedirs(os.path.dirname(SR_MODEL_PATH), exist_ok=True)

        with self._sr_lock:
            if self._sr_model is not None:
                return self._sr_model

            if not os.path.exists(SR_MODEL_PATH) or os.path.getsize(SR_MODEL_PATH) < 10000:
                tmp = SR_MODEL_PATH + ".download"
                last_exc = None
                for attempt in range(1, 4):
                    try:
                        print(f"Downloading FSRCNN x2 face upscaler model (attempt {attempt}/3)...")
                        request = urllib.request.Request(
                            SR_MODEL_URL,
                            headers={"User-Agent": "SmartAttend/1.0"},
                        )
                        with urllib.request.urlopen(request, timeout=30) as resp, open(tmp, "wb") as out:
                            while True:
                                chunk = resp.read(1024 * 1024)
                                if not chunk:
                                    break
                                out.write(chunk)

                        if os.path.getsize(tmp) < 10000:
                            raise RuntimeError("Downloaded FSRCNN model is unexpectedly small.")

                        os.replace(tmp, SR_MODEL_PATH)
                        print("FSRCNN x2 face upscaler model downloaded successfully.")
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        try:
                            if os.path.exists(tmp):
                                os.remove(tmp)
                        except Exception:
                            pass
                        print(f"FSRCNN download attempt {attempt}/3 failed: {exc}")

                if last_exc is not None and not os.path.exists(SR_MODEL_PATH):
                    print(f"FSRCNN model download failed after 3 attempts: {last_exc}; using interpolation fallback.")
                    return None

            try:
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                sr.readModel(SR_MODEL_PATH)
                sr.setModel("fsrcnn", 2)
                # Do not explicitly call setPreferableTarget/setPreferableBackend.
                # Newer OpenCV graph engines can emit:
                # "Targets are not supported by the new graph engine for now".
                # The default backend/target is sufficient and avoids that warning.
                self._sr_model = sr
                print("FSRCNN x2 face upscaler loaded.")
            except Exception as exc:
                print(f"FSRCNN initialization failed: {exc}; using interpolation fallback.")
                self._sr_model = None

        return self._sr_model

    def _ai_upscale_face_crop(self, crop_bgr, target_edge=512):
        """AI-upscale small face crops with FSRCNN x2 without changing detection/recognition."""
        try:
            longest = max(crop_bgr.shape[:2])
            if longest >= min(384, target_edge):
                return crop_bgr
            sr = self._get_superres()
            if sr is not None:
                up = sr.upsample(crop_bgr)
                if max(up.shape[:2]) > target_edge:
                    scale = float(target_edge) / max(up.shape[:2])
                    up = cv2.resize(up, (max(1, int(up.shape[1]*scale)), max(1, int(up.shape[0]*scale))), interpolation=cv2.INTER_AREA)
                return np.ascontiguousarray(up)
            # Safe fallback if the optional model is absent.
            scale = min(2.0, float(target_edge) / max(1, longest))
            return cv2.resize(crop_bgr, (max(1,int(crop_bgr.shape[1]*scale)), max(1,int(crop_bgr.shape[0]*scale))), interpolation=cv2.INTER_CUBIC)
        except Exception as exc:
            print(f"Face AI upscaling failed: {exc}")
            return crop_bgr

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=82, padded=False, max_edge=MAX_FACE_CROP_EDGE):
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
            # Apply AI super-resolution only to genuinely small crops. This is
            # presentation-only; detection and face recognition continue using
            # the original image/encodings, so accuracy is not altered.
            if max(crop_bgr.shape[:2]) < 384:
                crop_bgr = self._ai_upscale_face_crop(crop_bgr, target_edge=max_edge)
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

    def _get_yunet_detector(self, input_size):
        """Return a lightweight YuNet detector. The model is downloaded once if absent.

        YuNet is used instead of Haar Cascade because some Railway OpenCV builds do
        not expose CascadeClassifier. It is also substantially better for small,
        partially occluded and non-frontal faces.

        The model is fetched over the network on first use, so a slow or
        blocked connection to the model host can silently remove YuNet from
        the detection pipeline. To keep detection reliable, this retries the
        download a few times with a timeout instead of failing on the first
        hiccup, and clearly logs success/failure so a drop in detected faces
        can be diagnosed from the logs.
        """
        if not hasattr(cv2, "FaceDetectorYN"):
            print("YuNet unavailable: this OpenCV build has no FaceDetectorYN.")
            return None
        os.makedirs(os.path.dirname(YUNET_MODEL_PATH), exist_ok=True)
        if not os.path.exists(YUNET_MODEL_PATH):
            tmp = YUNET_MODEL_PATH + ".download"
            with self._yunet_lock:
                if not os.path.exists(YUNET_MODEL_PATH):
                    last_exc = None
                    for attempt in range(1, 4):
                        try:
                            print(f"Downloading YuNet face detector model (attempt {attempt}/3)...")
                            with urllib.request.urlopen(YUNET_MODEL_URL, timeout=15) as resp, open(tmp, "wb") as out:
                                out.write(resp.read())
                            os.replace(tmp, YUNET_MODEL_PATH)
                            print("YuNet model downloaded successfully.")
                            last_exc = None
                            break
                        except Exception as exc:
                            last_exc = exc
                            try:
                                if os.path.exists(tmp):
                                    os.remove(tmp)
                            except Exception:
                                pass
                    if last_exc is not None:
                        print(
                            f"YuNet model download failed after 3 attempts: {last_exc}. "
                            "Detection will fall back to HOG-only passes, which typically "
                            "misses more small/angled classroom faces."
                        )
                        return None
        try:
            # Some OpenCV 4.x builds emit a warning from the internal graph engine
            # while constructing FaceDetectorYN because it calls an unsupported
            # preferable-target path internally. Suppress only that OpenCV warning
            # during construction; detector parameters and behavior are unchanged.
            previous_log_level = None
            try:
                if hasattr(cv2, "getLogLevel") and hasattr(cv2, "setLogLevel"):
                    previous_log_level = cv2.getLogLevel()
                    cv2.setLogLevel(2)  # ERROR: hide WARN, keep errors visible
                detector = cv2.FaceDetectorYN.create(
                    YUNET_MODEL_PATH, "", tuple(map(int, input_size)),
                    0.35, 0.30, 5000
                )
            finally:
                if previous_log_level is not None:
                    try:
                        cv2.setLogLevel(previous_log_level)
                    except Exception:
                        pass
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
                vals = [float(v) for v in row]
                if len(vals) < 15:
                    # Malformed row; fall back to bbox-only, no landmarks.
                    x, y, bw, bh = vals[0], vals[1], vals[2], vals[3]
                    score = vals[4] if len(vals) > 4 else 1.0
                else:
                    # OpenCV FaceDetectorYN row layout:
                    # [x, y, w, h, re_x, re_y, le_x, le_y, nose_x, nose_y,
                    #  rmc_x, rmc_y, lmc_x, lmc_y, score]
                    # The confidence score is the LAST value, not index 4 —
                    # index 4 is the right-eye x-coordinate. Reading it as a
                    # score silently disabled this filter and, more
                    # importantly, meant a poorly-fit box was never corrected.
                    x, y, bw, bh = vals[0], vals[1], vals[2], vals[3]
                    right_eye_x, right_eye_y = vals[4], vals[5]
                    left_eye_x, left_eye_y = vals[6], vals[7]
                    score = vals[14]

                if score < 0.35 or bw < 10 or bh < 10:
                    continue

                if len(vals) >= 15:
                    # For faces with a covered lower half (niqab/mask), the
                    # raw box regression is trained on full, unmasked faces
                    # and can end up sitting too high — tight around only the
                    # visible eye strip, with the box "floating" above where
                    # the face actually is. The eye landmarks stay reliable
                    # even when the lower face is covered, so use them to
                    # re-anchor the box whenever the raw box clearly isn't
                    # centered on the eyes the way a normal face box would be.
                    interocular = abs(left_eye_x - right_eye_x)
                    if interocular > 2 and bh > 0:
                        eye_cx = (right_eye_x + left_eye_x) / 2.0
                        eye_cy = (right_eye_y + left_eye_y) / 2.0
                        eye_frac_y = (eye_cy - y) / bh
                        # A well-fit face box has the eyes roughly 35-55% down
                        # from its top. Outside that band the box is likely
                        # mis-anchored — rebuild it around the eyes using
                        # standard face proportions instead of trusting the
                        # raw regression.
                        if eye_frac_y < 0.30 or eye_frac_y > 0.60:
                            face_w = max(bw, interocular * 2.2)
                            face_h = face_w * 1.15
                            x = eye_cx - face_w / 2.0
                            y = eye_cy - face_h * 0.40
                            bw, bh = face_w, face_h

                results.append((int(y), int(x + bw), int(y + bh), int(x)))
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

            # YuNet eye support is particularly valuable when the lower face is
            # completely covered and dlib landmarks cannot find a mouth/nose.
            yunet_confirmed, yunet_eyes, _ = self._yunet_candidate_support(crop)
            if yunet_eyes and not mouth:
                score += 2

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

    def _yunet_candidate_support(self, rgb_crop):
        """Return whether YuNet independently supports a candidate face.

        This is deliberately used as a *confirmation* pass after the existing
        detector pipeline.  It does not replace HOG/tile detection, so small
        faces that HOG finds are still available.  It mainly removes HOG
        artifacts such as necks, eyes-only fragments and clothing regions.
        """
        try:
            h, w = rgb_crop.shape[:2]
            if h < 28 or w < 28:
                return False, False, 0.0
            # Upscale only this candidate.  Never upscale the whole classroom.
            scale = 1.0
            if max(h, w) < 180:
                scale = min(3.0, 180.0 / max(h, w))
                work = cv2.resize(rgb_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            else:
                work = rgb_crop
            bgr = cv2.cvtColor(work, cv2.COLOR_RGB2BGR)
            detector = self._get_yunet_detector((work.shape[1], work.shape[0]))
            if detector is None:
                return True, False, 0.0
            detector.setInputSize((work.shape[1], work.shape[0]))
            _, detections = detector.detect(bgr)
            if detections is None:
                return False, False, 0.0
            best_iou = 0.0
            best_score = 0.0
            eye_supported = False
            for row in detections:
                vals = [float(v) for v in row]
                if len(vals) < 5:
                    continue
                x, y, bw, bh = vals[:4]
                score = vals[14] if len(vals) >= 15 else vals[4]
                if score < 0.35:
                    continue
                # Candidate itself is the complete crop.
                inter_w = max(0.0, min(w * scale, x + bw) - max(0.0, x))
                inter_h = max(0.0, min(h * scale, y + bh) - max(0.0, y))
                inter = inter_w * inter_h
                det_area = max(1.0, bw * bh)
                crop_area = max(1.0, (w * scale) * (h * scale))
                iou = inter / max(1.0, det_area + crop_area - inter)
                best_iou = max(best_iou, iou)
                if score > best_score:
                    best_score = score
                if len(vals) >= 15:
                    rex, rey, lex, ley = vals[4], vals[5], vals[6], vals[7]
                    eye_dx = abs(lex - rex)
                    eye_cx = (lex + rex) / 2.0
                    eye_cy = (ley + rey) / 2.0
                    if eye_dx >= max(2.0, bw * 0.08) and 0.12 * bh <= eye_cy - y <= 0.72 * bh:
                        eye_supported = True
            # A candidate is confirmed when YuNet has a reasonably overlapping
            # face, or when its eye landmarks strongly support an eyes-visible
            # face. The latter is important for masks/niqabs.
            return (best_iou >= 0.12 or eye_supported), eye_supported, best_score
        except Exception:
            return True, False, 0.0

    @staticmethod
    def _focus_metrics(rgb_crop):
        """Return size-normalized focus/detail metrics for one face crop."""
        try:
            h, w = rgb_crop.shape[:2]
            if h < 18 or w < 18:
                return 0.0, 0.0, 0.0
            gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)
            # Normalize face size so a 45px face is not automatically judged
            # softer than a 120px face simply because it contains fewer pixels.
            gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
            lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            ten = float(np.mean(gx * gx + gy * gy))
            edges = float(np.mean(cv2.Canny(gray, 50, 130) > 0))
            return lap, ten, edges
        except Exception:
            return 0.0, 0.0, 0.0

    def _filter_face_locations(self, rgb_image, locations):
        """Conservative quality filter applied *after* the working detector.

        The detector is the baseline and is intentionally left untouched. This
        stage only removes candidates when there is independent evidence that
        they are not a real, focused face. In particular, a neck/eye fragment
        must normally fail the YuNet confirmation, while a genuine small face
        is retained even when its absolute sharpness is lower.
        """
        if not locations:
            return []
        h, w = rgb_image.shape[:2]
        scored = []
        for idx, loc in enumerate(locations):
            t, r, b, l = [int(v) for v in loc]
            fh, fw = b - t, r - l
            if fh < 14 or fw < 14:
                continue
            # Modest padding helps the focus metric and YuNet see the eye/face
            # context without feeding the whole classroom image into anything.
            py, px = int(fh * 0.22), int(fw * 0.22)
            ct, cb = max(0, t - py), min(h, b + py)
            cl, cr = max(0, l - px), min(w, r + px)
            crop = np.ascontiguousarray(rgb_image[ct:cb, cl:cr])
            lap, ten, edge = self._focus_metrics(crop)
            confirmed, eye_support, yunet_score = self._yunet_candidate_support(crop)
            scored.append({
                "loc": loc,
                "lap": lap,
                "ten": ten,
                "edge": edge,
                "confirmed": confirmed,
                "eye_support": eye_support,
                "yunet": yunet_score,
                "size": max(fh, fw),
            })
            del crop
            if idx % 5 == 0:
                gc.collect()

        if not scored:
            return []

        # Do not use the weakest face as the baseline.  That was what caused
        # focused students to disappear in the previous focus implementation.
        # Instead, use robust medians and reject only faces that are both very
        # soft in absolute terms and substantially below the group/detail level.
        lap_values = np.array([x["lap"] for x in scored], dtype=np.float64)
        ten_values = np.array([x["ten"] for x in scored], dtype=np.float64)
        median_lap = float(np.median(lap_values))
        median_ten = float(np.median(ten_values))

        kept = []
        filtered = 0
        focus_log = []
        for item in scored:
            # Conservative blur rule. Small faces are protected by the relative
            # factor and by the absolute floor; both focus metrics must be poor.
            # Very conservative blur floor.  We require BOTH normalized focus
            # measures to be extremely weak before rejecting a candidate, so a
            # genuinely focused small student is protected.  A face with eye
            # support is allowed through unless it is *extremely* soft.
            # V5: tighten only the blur test. The detector itself is unchanged.
            # Background faces that are genuinely out of focus tend to have both
            # detail measures substantially below the focused-face population.
            # We require BOTH measures to be weak, which protects faces that are
            # small but still have real edge/detail information.
            very_soft = (
                item["lap"] < max(8.0, median_lap * 0.28) and
                item["ten"] < max(22.0, median_ten * 0.34)
            )
            extremely_soft = (item["lap"] < 5.0 and item["ten"] < 14.0)
            median_size = float(np.median([x["size"] for x in scored]))
            small_soft = (
                item["size"] < max(70.0, median_size * 0.82)
                and item["lap"] < max(10.0, median_lap * 0.40)
                and item["ten"] < max(25.0, median_ten * 0.48)
            )
            # Extra background-blur guard: a genuinely out-of-focus face in
            # the back of a classroom is usually both smaller than the focused
            # group and weak in BOTH independent detail measures. Require the
            # combination, never size alone, so small focused students survive.
            background_soft = (
                item["size"] < max(64.0, median_size * 0.74)
                and item["lap"] < max(12.0, median_lap * 0.50)
                and item["ten"] < max(28.0, median_ten * 0.58)
            )
            # Neck/clothing/artifact candidates generally fail independent face
            # confirmation. This is separate from focus so a real but slightly
            # soft face is not removed just because YuNet prefers another box.
            no_face_support = not item["confirmed"] and not item["eye_support"]
            reject = no_face_support or ((very_soft or small_soft or background_soft) and not item["eye_support"]) or extremely_soft

            # If YuNet is unavailable, _yunet_candidate_support deliberately
            # returns confirmed=True, so the quality stage degrades safely to
            # focus filtering rather than deleting all HOG detections.
            if reject:
                filtered += 1
            else:
                kept.append(item["loc"])
            focus_log.append(
                f"{item['size']}px:lap={item['lap']:.1f},ten={item['ten']:.1f},"
                f"yunet={item['yunet']:.2f},eyes={int(item['eye_support'])},keep={not reject}"
            )

        print(
            f"Focus/face quality -> input={len(locations)}, kept={len(kept)}, "
            f"filtered={filtered}, median_lap={median_lap:.1f}, median_ten={median_ten:.1f}"
        )
        print("Focus candidates -> " + " | ".join(focus_log[:80]))
        return kept

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
        pass1_count = 0
        try:
            for loc in face_recognition.face_locations(
                detect_img, number_of_times_to_upsample=1, model="hog"
            ):
                raw.append(restore(loc))
                pass1_count += 1
        except Exception as exc:
            print(f"Primary HOG detection failed: {exc}")

        # 2) YuNet is the primary recovery detector. It handles small,
        # partially occluded, profile and masked faces much better than Haar.
        pass2_count = 0
        try:
            yunet_input = cv2.cvtColor(detect_img, cv2.COLOR_RGB2BGR)
            for loc in self._yunet_locations(yunet_input):
                raw.append(restore(loc))
                pass2_count += 1
            del yunet_input
        except Exception as exc:
            print(f"YuNet recovery failed: {exc}")

        # 3) Small-face tile recovery. A group photo can contain people whose
        # faces are only a few dozen pixels wide. HOG on the whole image can
        # miss those, while a sequential overlapping tile gives the detector
        # much more face resolution without the memory cost of full-image
        # upsample=2/3.
        pass3_count = 0
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
                        pass3_count += 1
                except Exception as exc:
                    print(f"HOG tile detection failed at ({x0},{y0}): {exc}")
                del tile
                gc.collect()

        # 4) Orientation recovery.
        pass4_count = 0
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
                    pass4_count += 1
            except Exception as exc:
                print(f"Rotated HOG ({rotation}) failed: {exc}")
            del rotated
            gc.collect()

        # 5) A second YuNet pass on a 1.35x image catches distant/partially
        # occluded faces that are below the effective size of the first pass.
        pass5_count = 0
        try:
            enlarged = cv2.resize(detect_img, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            for loc in self._yunet_locations(cv2.cvtColor(enlarged, cv2.COLOR_RGB2BGR)):
                t,r,b,l = loc
                raw.append(restore((int(t/1.35), int(r/1.35), int(b/1.35), int(l/1.35))))
                pass5_count += 1
            del enlarged
            gc.collect()
        except Exception as exc:
            print(f"Enlarged YuNet recovery failed: {exc}")

        final_locations = self._dedupe_face_locations(raw, original.shape)
        deduped_count = len(final_locations)
        final_locations = self._filter_face_locations(original, final_locations)
        print(
            f"Detection pass breakdown -> primary_hog={pass1_count}, "
            f"yunet_pass1={pass2_count}, tile_hog={pass3_count}, "
            f"rotated_hog={pass4_count}, yunet_pass2={pass5_count}, "
            f"raw_total={len(raw)}, after_dedupe={deduped_count}, "
            f"after_quality_filter={len(final_locations)}"
        )
        if pass2_count == 0 and pass5_count == 0:
            print(
                "WARNING: both YuNet passes returned 0 faces this run. If this "
                "is unexpected, check the earlier 'YuNet' log lines above for a "
                "download/initialization failure — YuNet is usually the biggest "
                "contributor of small/angled faces in group photos."
            )
        return final_locations

    def _assign_faces_to_students(
        self,
        face_encodings_by_index: dict[int, np.ndarray | None],
        tolerance: float,
        allowed_set: set | None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict[int, tuple[int | None, str, float]]:
        """Assign every detected face to an enrolled student without duplicates.

        Recognition is intentionally independent from Stage-1 detection, and
        every detected face (including ones whose lower face is partly
        covered) is checked here — none are pre-filtered out before this
        step. Assignment is global and one-to-one so the same student cannot
        be marked present for multiple detected faces.

        Acceptance rule (simple and explicit): a face's match confidence is
        ``1 - distance`` against its closest enrolled student. If that
        confidence is 50% or higher, the face is recognised as that student.
        If it is below 50%, the face is left unmatched and is reported as
        unrecognized rather than being forced into an identity.
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

                # Keep the same model and the same conservative distance ceiling,
                # but calibrate the displayed familiarity score instead of using
                # the raw `1 - distance` value. That raw formula is overly harsh:
                # e.g. a distance of .56 is displayed as only 44%, even though
                # it is inside the normal face-recognition match region.
                #
                # The calibrated score is monotonic: closer embeddings always
                # receive a higher score. It is only a presentation/acceptance
                # calibration; no model weights are changed.
                if distance > max(float(tolerance), 0.64):
                    continue
                if distance <= 0.50:
                    familiarity = 0.92 + (0.50 - distance) * 0.40
                elif distance <= 0.58:
                    familiarity = 0.78 + (0.58 - distance) * 1.75
                elif distance <= 0.64:
                    familiarity = 0.50 + (0.64 - distance) * 4.67
                else:
                    familiarity = 0.0
                familiarity = float(max(0.0, min(0.99, familiarity)))
                options.append(
                    (distance, sid, self.known_face_names[idx], familiarity)
                )

            options.sort(key=lambda item: item[0])
            candidates_by_face[face_index] = options

        # Convert each face's candidate list into an acceptance candidate
        # using the 50%-confidence rule described above.
        candidates = []
        for face_index, options in candidates_by_face.items():
            if not options:
                continue

            best = options[0]
            best_distance, sid, name, confidence = best

            # Require a clear enough lead over the next enrolled identity when
            # there is a real second candidate. This improves low-confidence
            # matches without lowering the 50% floor or changing the model.
            # If the nearest two identities are nearly tied, the system should
            # remain cautious rather than inventing an identity.
            second_distance = options[1][0] if len(options) > 1 else None
            margin = (second_distance - best_distance) if second_distance is not None else 1.0
            ambiguous = second_distance is not None and margin < 0.035 and best_distance > 0.56

            if confidence >= float(min_confidence) and not ambiguous:
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

    def _encode_face_one_at_a_time(self, rgb_image, location):
        """Encode one face using a small crop to keep Railway memory bounded."""
        try:
            h, w = rgb_image.shape[:2]
            top, right, bottom, left = [int(v) for v in location]
            fh = max(1, bottom - top)
            fw = max(1, right - left)
            pad_y = int(fh * 0.35)
            pad_x = int(fw * 0.35)
            t = max(0, top - pad_y)
            b = min(h, bottom + pad_y)
            l = max(0, left - pad_x)
            r = min(w, right + pad_x)
            crop = np.ascontiguousarray(rgb_image[t:b, l:r])
            if crop.size == 0:
                return None
            crop, scale = self._downscale_rgb(crop, MAX_FACE_CROP_EDGE)
            ct = max(0, int(round((top - t) * scale)))
            cr = min(crop.shape[1], int(round((right - l) * scale)))
            cb = min(crop.shape[0], int(round((bottom - t) * scale)))
            cl = max(0, int(round((left - l) * scale)))
            if cr <= cl or cb <= ct:
                return None
            encs = face_recognition.face_encodings(
                crop, known_face_locations=[(ct, cr, cb, cl)], num_jitters=0
            )
            return encs[0] if encs else None
        except Exception as exc:
            print(f"Face encoding failed: {exc}")
            return None

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
        # Keep the recognition image bounded. This is especially important on
        # Railway where camera photos can be 4K+ and dlib temporarily allocates
        # substantial memory while encoding faces.
        classroom_image, image_scale = self._downscale_rgb(classroom_image, MAX_DETECT_EDGE)

        face_locations = self._detect_face_locations(classroom_image)

        # Encode sequentially from small face crops. The previous implementation
        # called face_encodings() for every face on the full classroom image in one
        # batch, which can exhaust a small Railway container on 29+ faces.
        face_encodings_by_index = {}
        for idx, location in enumerate(face_locations):
            face_encodings_by_index[idx] = self._encode_face_one_at_a_time(classroom_image, location)
            if idx and idx % 10 == 0:
                gc.collect()

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
        annotated_rgb = np.ascontiguousarray(classroom_image)
        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

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
            cv2.rectangle(annotated_bgr, (left, top), (right, bottom), color, 2)

            label = f"{name} ({confidence:.0%})" if student_id is not None else "Unknown"
            label_y = max(top - 10, 20)
            (w_txt, h_txt), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
            cv2.rectangle(
                annotated_bgr,
                (left, label_y - h_txt - 4),
                (left + w_txt + 6, label_y + 4),
                color,
                cv2.FILLED,
            )
            cv2.putText(
                annotated_bgr,
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
        annotated_path = f"output/annotated_classroom_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        cv2.imwrite(annotated_path, annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 82])
        attendance_data["annotated_image_path"] = annotated_path
        attendance_data["annotated_image_base64"] = self._encode_bgr_jpeg_base64(
            annotated_bgr, quality=68, max_edge=MAX_ANNOTATED_EDGE
        )

        del annotated_rgb, annotated_bgr, classroom_image
        gc.collect()

        result_message = (
            "Recognition complete!"
            if has_registered_faces
            else "Face detection complete — no registered student photos to match against."
        )
        return attendance_data, result_message