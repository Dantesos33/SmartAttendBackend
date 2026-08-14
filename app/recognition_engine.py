import face_recognition
import cv2
import numpy as np
import os
import base64
import gc
from datetime import datetime
import json

MIN_CONFIDENCE = 0.45
MAX_DETECT_EDGE = 2000
MAX_ANNOTATED_EDGE = 1200
# A second, higher-sensitivity HOG pass is used when the first pass may have
# missed smaller/background faces in group photos. Keeping this pass on a
# smaller image prevents the upsample=2 operation from becoming excessive.
SECOND_PASS_DETECT_EDGE = 1000
SECOND_PASS_UPSAMPLE = 2


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

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=85):
        try:
            h, w = rgb_image.shape[:2]
            top = max(0, int(top))
            left = max(0, int(left))
            bottom = min(h, int(bottom))
            right = min(w, int(right))
            if bottom <= top or right <= left:
                return None
            crop = rgb_image[top:bottom, left:right]
            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            return self._encode_bgr_jpeg_base64(crop_bgr, quality=quality)
        except Exception:
            return None

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
        top = max(at, bt)
        left = max(al, bl)
        bottom = min(ab, bb)
        right = min(ar, br)
        inter_w = max(0, right - left)
        inter_h = max(0, bottom - top)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ar - al) * max(0, ab - at)
        area_b = max(0, br - bl) * max(0, bb - bt)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    def _scale_locations_to_original(self, locations, scale, width, height):
        inv_scale = 1.0 / scale
        result = []
        for top, right, bottom, left in locations:
            orig_top = max(0, int(round(top * inv_scale)))
            orig_right = min(width, int(round(right * inv_scale)))
            orig_bottom = min(height, int(round(bottom * inv_scale)))
            orig_left = max(0, int(round(left * inv_scale)))
            if orig_bottom > orig_top + 15 and orig_right > orig_left + 15:
                result.append((orig_top, orig_right, orig_bottom, orig_left))
        return result

    @staticmethod
    def _box_iou(a, b):
        at, ar, ab, al = a
        bt, br, bb, bl = b
        top = max(at, bt)
        left = max(al, bl)
        bottom = min(ab, bb)
        right = min(ar, br)
        inter_w = max(0, right - left)
        inter_h = max(0, bottom - top)
        inter = inter_w * inter_h
        if inter == 0:
            return 0.0
        area_a = max(0, ar - al) * max(0, ab - at)
        area_b = max(0, br - bl) * max(0, bb - bt)
        union = area_a + area_b - inter
        return inter / union if union else 0.0

    @staticmethod
    def _rotate_image(rgb_image, angle):
        """Rotate an RGB image without changing its content size unexpectedly."""
        if angle == 90:
            return cv2.rotate(rgb_image, cv2.ROTATE_90_CLOCKWISE)
        if angle == -90:
            return cv2.rotate(rgb_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if angle == 180:
            return cv2.rotate(rgb_image, cv2.ROTATE_180)

        h, w = rgb_image.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = abs(matrix[0, 0])
        sin = abs(matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        matrix[0, 2] += (new_w / 2.0) - center[0]
        matrix[1, 2] += (new_h / 2.0) - center[1]
        return cv2.warpAffine(
            rgb_image,
            matrix,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    @staticmethod
    def _map_rotated_box_to_original(box, angle, original_shape):
        """Map a box from a rotated image back to original top/right/bottom/left."""
        top, right, bottom, left = box
        h, w = original_shape[:2]
        corners = np.array(
            [[left, top], [right, top], [right, bottom], [left, bottom]],
            dtype=np.float32,
        )

        if angle == 90:
            # Inverse of clockwise rotation: x = y', y = h - x'.
            mapped = np.column_stack((corners[:, 1], h - corners[:, 0]))
        elif angle == -90:
            # Inverse of counter-clockwise rotation: x = w - y', y = x'.
            mapped = np.column_stack((w - corners[:, 1], corners[:, 0]))
        elif angle == 180:
            mapped = np.column_stack((w - corners[:, 0], h - corners[:, 1]))
        else:
            # For arbitrary rotations the detector is only used as a fallback.
            # Re-running on a rotated image is still useful, but exact inverse
            # mapping is required for the resulting rectangle.
            radians = np.deg2rad(-angle)
            c, s = np.cos(radians), np.sin(radians)
            center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
            rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
            mapped = (corners - center) @ rotation.T + center

        xs = mapped[:, 0]
        ys = mapped[:, 1]
        return (
            max(0, int(round(ys.min()))),
            min(w, int(round(xs.max()))),
            min(h, int(round(ys.max()))),
            max(0, int(round(xs.min()))),
        )

    def _scale_locations_to_original(self, locations, scale, width, height):
        inv_scale = 1.0 / scale
        result = []
        for top, right, bottom, left in locations:
            orig_top = max(0, int(round(top * inv_scale)))
            orig_right = min(width, int(round(right * inv_scale)))
            orig_bottom = min(height, int(round(bottom * inv_scale)))
            orig_left = max(0, int(round(left * inv_scale)))
            if orig_bottom > orig_top + 15 and orig_right > orig_left + 15:
                result.append((orig_top, orig_right, orig_bottom, orig_left))
        return result

    def _detect_face_locations(self, rgb_image):
        """Multi-pass detector for group photos and non-upright faces.

        HOG remains the fast primary detector. We then add a smaller, upsampled
        HOG pass, OpenCV frontal/profile cascades, and rotated HOG passes. The
        rotated passes are important for phones/photos where a person's head is
        tilted sideways or the whole image is slightly rotated.
        """
        h, w = rgb_image.shape[:2]
        detect_img, scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        primary = face_recognition.face_locations(
            detect_img, number_of_times_to_upsample=1, model="hog"
        )
        full_locations = self._scale_locations_to_original(primary, scale, w, h)

        def add_boxes(boxes, iou_threshold=0.30):
            for box in boxes:
                if not any(self._box_iou(box, existing) >= iou_threshold for existing in full_locations):
                    full_locations.append(box)

        # Higher sensitivity for small/far-away faces.
        if len(full_locations) < 6:
            sensitive_img, sensitive_scale = self._downscale_rgb(rgb_image, SECOND_PASS_DETECT_EDGE)
            try:
                secondary = face_recognition.face_locations(
                    sensitive_img, number_of_times_to_upsample=SECOND_PASS_UPSAMPLE, model="hog"
                )
                add_boxes(
                    self._scale_locations_to_original(secondary, sensitive_scale, w, h),
                    iou_threshold=0.35,
                )
            except Exception as e:
                print(f"Secondary face detection skipped: {e}")

        # Haar cascades are deliberately used as a candidate generator rather than
        # the primary detector. They are very good at recovering frontal/profile
        # faces that HOG occasionally misses, but can also produce body/background
        # false positives, so candidates are size/position filtered and merged.
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            cascade_files = (
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_profileface.xml",
            )
            for cascade_file in cascade_files:
                cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, cascade_file))
                if cascade.empty():
                    continue
                boxes = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=5 if "profile" not in cascade_file else 4,
                    minSize=(40, 40),
                )
                candidates = []
                for x, y, box_w, box_h in boxes:
                    center_y = y + box_h / 2.0
                    aspect = box_w / float(max(box_h, 1))
                    # Classroom faces are expected in the upper/middle area. Keep
                    # a generous range so seated/shorter students are not excluded.
                    if center_y > h * 0.78:
                        continue
                    if box_w < 40 or box_h < 40 or box_w > w * 0.30 or box_h > h * 0.30:
                        continue
                    if aspect < 0.60 or aspect > 1.65:
                        continue
                    candidates.append((y, x + box_w, y + box_h, x))
                add_boxes(candidates, iou_threshold=0.25)
        except Exception as e:
            print(f"OpenCV face-detection fallback skipped: {e}")

        # Run HOG on rotated versions only when the normal detector is still likely
        # incomplete. This catches tilted/sideways faces without making every
        # classroom capture pay the cost of several extra passes.
        if len(full_locations) < 4:
            for angle in (90, -90, 180):
                try:
                    rotated = self._rotate_image(rgb_image, angle)
                    rotated_small, rotated_scale = self._downscale_rgb(rotated, 1400)
                    rotated_locations = face_recognition.face_locations(
                        rotated_small, number_of_times_to_upsample=1, model="hog"
                    )
                    # First map from rotated-small coordinates back to rotated image,
                    # then map those coordinates back to the original image.
                    rotated_full = self._scale_locations_to_original(
                        rotated_locations, rotated_scale, rotated.shape[1], rotated.shape[0]
                    )
                    mapped = [
                        self._map_rotated_box_to_original(box, angle, rgb_image.shape)
                        for box in rotated_full
                    ]
                    add_boxes(mapped, iou_threshold=0.25)
                except Exception as e:
                    print(f"Rotated HOG pass ({angle}°) skipped: {e}")

        full_locations.sort(key=lambda box: (box[0], box[3]))
        print(f"Detected {len(full_locations)} face(s) in image.")
        return full_locations

    def _orientation_fallback_encoding(self, rgb_image, location):
        """Try to encode a difficult face after normalizing its orientation.

        This is intentionally a fallback for faces that HOG/landmark encoding cannot
        match. It avoids running expensive multi-angle encoding on every face.
        """
        top, right, bottom, left = location
        h, w = rgb_image.shape[:2]
        pad_y = max(12, int((bottom - top) * 0.45))
        pad_x = max(12, int((right - left) * 0.45))
        top = max(0, top - pad_y)
        left = max(0, left - pad_x)
        bottom = min(h, bottom + pad_y)
        right = min(w, right + pad_x)
        crop = np.ascontiguousarray(rgb_image[top:bottom, left:right])
        if crop.size == 0:
            return None

        best = None
        for angle in (0, 90, -90, 180):
            try:
                rotated = self._rotate_image(crop, angle)
                rotated, _ = self._downscale_rgb(rotated, 700)
                locations = face_recognition.face_locations(
                    rotated, number_of_times_to_upsample=1, model="hog"
                )
                if not locations:
                    continue
                # Pick the largest detected face in the padded crop.
                chosen = max(
                    locations,
                    key=lambda b: max(0, b[2] - b[0]) * max(0, b[1] - b[3]),
                )
                encodings = face_recognition.face_encodings(
                    rotated, [chosen], num_jitters=2
                )
                if encodings:
                    best = encodings[0]
                    if angle == 0:
                        break
            except Exception:
                continue
        return best

    def _assign_faces_to_students(
        self,
        face_encodings_by_index: dict[int, np.ndarray | None],
        tolerance: float,
        allowed_set: set | None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict[int, tuple[int | None, str, float]]:
        candidates = []

        for face_index, encoding in face_encodings_by_index.items():
            if encoding is None or not self.known_face_encodings:
                continue
            distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            for idx in np.argsort(distances):
                distance = float(distances[idx])
                if distance > tolerance:
                    break
                student_id = self.known_face_ids[idx]
                if allowed_set is not None and student_id not in allowed_set:
                    continue
                confidence = 1.0 - distance
                if confidence < min_confidence:
                    continue
                candidates.append((face_index, student_id, self.known_face_names[idx], confidence, distance))

        candidates.sort(key=lambda item: item[4])

        assigned_faces = set()
        assigned_students = set()
        assignments = {}

        for face_index, student_id, name, confidence, _distance in candidates:
            if face_index in assigned_faces or student_id in assigned_students:
                continue
            assignments[face_index] = (student_id, name, confidence)
            assigned_faces.add(face_index)
            assigned_students.add(student_id)

        for face_index in face_encodings_by_index:
            if face_index not in assignments:
                assignments[face_index] = (None, "Unknown", 0.0)

        return assignments

    def recognize_classroom(
        self,
        classroom_image_path,
        tolerance=0.55,
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

            # If a detected face was not recognized, retry that face using a
            # padded crop rotated through 0/90/-90/180 degrees. This is the
            # important second stage for sideways/tilted/downward-looking faces:
            # detection and recognition are separate problems, and a good box can
            # still produce poor landmarks when the head is not upright.
            difficult_faces = [
                idx for idx, assignment in assignments.items()
                if assignment[0] is None
            ]
            for face_index in difficult_faces:
                location = face_locations[face_index]
                fallback_encoding = self._orientation_fallback_encoding(
                    classroom_image, location
                )
                if fallback_encoding is None:
                    continue
                retry = self._assign_faces_to_students(
                    {face_index: fallback_encoding},
                    tolerance=tolerance,
                    allowed_set=allowed_set,
                )
                if retry.get(face_index, (None, "Unknown", 0.0))[0] is not None:
                    candidate = retry[face_index]
                    # Do not replace a stronger match for another face with the
                    # same student. The normal global assignment remains primary.
                    already_used = any(
                        value[0] == candidate[0] and idx != face_index
                        for idx, value in assignments.items()
                    )
                    if not already_used:
                        assignments[face_index] = candidate

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