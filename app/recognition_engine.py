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

    def _detect_face_locations(self, rgb_image):
        """High-recall detector for classroom/group photos.

        The normal HOG detector is fast, but small faces near the edge of a
        group photo are easy to miss. We therefore combine:
          1. normal HOG,
          2. an upsampled HOG pass,
          3. OpenCV frontal/profile Haar cascades,
          4. overlapping tiles for small/background faces, and
          5. rotated passes for sideways/tilted faces.

        All detections are merged so the same face is only returned once.
        """
        h, w = rgb_image.shape[:2]
        full_locations = []

        def add_boxes(boxes, iou_threshold=0.30):
            for box in boxes:
                top, right, bottom, left = box
                if bottom <= top or right <= left:
                    continue
                if bottom - top < 24 or right - left < 24:
                    continue
                if not any(self._box_iou(box, existing) >= iou_threshold for existing in full_locations):
                    full_locations.append(box)

        # 1) Primary HOG pass.
        detect_img, scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        try:
            primary = face_recognition.face_locations(
                detect_img, number_of_times_to_upsample=1, model="hog"
            )
            add_boxes(self._scale_locations_to_original(primary, scale, w, h))
        except Exception as e:
            print(f"Primary face detection failed: {e}")

        # 2) Higher sensitivity HOG pass. This is especially useful for the
        # fourth person in the supplied 1200x900 image, whose face is only
        # roughly 70-80 pixels wide.
        if len(full_locations) < 6:
            try:
                sensitive, sensitive_scale = self._downscale_rgb(
                    rgb_image, SECOND_PASS_DETECT_EDGE
                )
                secondary = face_recognition.face_locations(
                    sensitive,
                    number_of_times_to_upsample=SECOND_PASS_UPSAMPLE,
                    model="hog",
                )
                add_boxes(
                    self._scale_locations_to_original(
                        secondary, sensitive_scale, w, h
                    ),
                    iou_threshold=0.30,
                )
            except Exception as e:
                print(f"Secondary face detection skipped: {e}")

        # 3) Haar frontal/profile detectors. They are candidate generators;
        # recognition still has to validate the detected face.
        try:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            cascade_specs = (
                ("haarcascade_frontalface_default.xml", 4),
                ("haarcascade_frontalface_alt2.xml", 4),
                ("haarcascade_profileface.xml", 3),
            )
            for cascade_file, neighbors in cascade_specs:
                cascade = cv2.CascadeClassifier(
                    os.path.join(cv2.data.haarcascades, cascade_file)
                )
                if cascade.empty():
                    continue
                boxes = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.05,
                    minNeighbors=neighbors,
                    minSize=(35, 35),
                )
                candidates = []
                for x, y, box_w, box_h in boxes:
                    # Ignore detections that are clearly too large to be a face
                    # or down in the floor/body area.
                    if box_w > w * 0.32 or box_h > h * 0.32:
                        continue
                    if y + box_h * 0.5 > h * 0.82:
                        continue
                    aspect = box_w / float(max(box_h, 1))
                    if aspect < 0.55 or aspect > 1.80:
                        continue
                    candidates.append((y, x + box_w, y + box_h, x))
                add_boxes(candidates, iou_threshold=0.25)
        except Exception as e:
            print(f"OpenCV face-detection fallback skipped: {e}")

        # 4) Overlapping tiles. Running HOG on an enlarged local tile makes a
        # small face become much larger relative to the detector's input.
        if len(full_locations) < 4:
            try:
                overlap = 0.22
                rows = 2
                cols = 2
                tile_h = int(h / (1.0 + overlap))
                tile_w = int(w / (1.0 + overlap))
                tile_h = min(h, max(tile_h, int(h * 0.55)))
                tile_w = min(w, max(tile_w, int(w * 0.55)))
                y_starts = sorted(set([0, max(0, h - tile_h)]))
                x_starts = sorted(set([0, max(0, w - tile_w)]))

                for y0 in y_starts:
                    for x0 in x_starts:
                        tile = np.ascontiguousarray(
                            rgb_image[y0:y0 + tile_h, x0:x0 + tile_w]
                        )
                        tile, tile_scale = self._downscale_rgb(tile, 1200)
                        locations = face_recognition.face_locations(
                            tile,
                            number_of_times_to_upsample=1,
                            model="hog",
                        )
                        mapped = []
                        inv = 1.0 / tile_scale
                        for top, right, bottom, left in locations:
                            mapped.append((
                                max(0, int(round(top * inv)) + y0),
                                min(w, int(round(right * inv)) + x0),
                                min(h, int(round(bottom * inv)) + y0),
                                max(0, int(round(left * inv)) + x0),
                            ))
                        add_boxes(mapped, iou_threshold=0.25)
            except Exception as e:
                print(f"Tiled face-detection pass skipped: {e}")

        # 5) Rotate the image for genuinely sideways/tilted faces. Haar is
        # also run on the rotated images because HOG is not reliably rotation
        # invariant.
        if len(full_locations) < 4:
            for angle in (90, -90, 180):
                try:
                    rotated = self._rotate_image(rgb_image, angle)
                    rotated_small, rotated_scale = self._downscale_rgb(rotated, 1400)
                    rotated_locations = face_recognition.face_locations(
                        rotated_small,
                        number_of_times_to_upsample=1,
                        model="hog",
                    )
                    rotated_full = self._scale_locations_to_original(
                        rotated_locations,
                        rotated_scale,
                        rotated.shape[1],
                        rotated.shape[0],
                    )
                    add_boxes(
                        [
                            self._map_rotated_box_to_original(
                                box, angle, rgb_image.shape
                            )
                            for box in rotated_full
                        ],
                        iou_threshold=0.25,
                    )

                    gray_rot = cv2.cvtColor(rotated, cv2.COLOR_RGB2GRAY)
                    cascade = cv2.CascadeClassifier(
                        os.path.join(
                            cv2.data.haarcascades,
                            "haarcascade_frontalface_default.xml",
                        )
                    )
                    boxes = cascade.detectMultiScale(
                        gray_rot,
                        scaleFactor=1.06,
                        minNeighbors=4,
                        minSize=(35, 35),
                    )
                    mapped = []
                    for x, y, bw, bh in boxes:
                        box = (y, x + bw, y + bh, x)
                        mapped.append(
                            self._map_rotated_box_to_original(
                                box, angle, rgb_image.shape
                            )
                        )
                    add_boxes(mapped, iou_threshold=0.25)
                except Exception as e:
                    print(f"Rotated detection pass ({angle}°) skipped: {e}")

        full_locations.sort(key=lambda box: (box[0], box[3]))
        print(f"Detected {len(full_locations)} face(s) in image.")
        return full_locations

    def _encode_face_robustly(self, rgb_image, location):
        """Encode one face without letting a difficult face break the batch."""
        try:
            encodings = face_recognition.face_encodings(
                rgb_image,
                known_face_locations=[location],
                num_jitters=1,
            )
            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"Direct face encoding failed: {e}")

        # Small Haar/HOG boxes benefit greatly from a local upscale.
        top, right, bottom, left = location
        h, w = rgb_image.shape[:2]
        pad_y = max(10, int((bottom - top) * 0.35))
        pad_x = max(10, int((right - left) * 0.35))
        top = max(0, top - pad_y)
        right = min(w, right + pad_x)
        bottom = min(h, bottom + pad_y)
        left = max(0, left - pad_x)
        crop = np.ascontiguousarray(rgb_image[top:bottom, left:right])
        if crop.size == 0:
            return None

        try:
            scale = max(2.0, 180.0 / max(1, bottom - top))
            scale = min(scale, 4.0)
            enlarged = cv2.resize(
                crop,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
            locations = face_recognition.face_locations(
                enlarged,
                number_of_times_to_upsample=1,
                model="hog",
            )
            if locations:
                chosen = max(
                    locations,
                    key=lambda b: max(0, b[2] - b[0]) * max(0, b[1] - b[3]),
                )
                encodings = face_recognition.face_encodings(
                    enlarged, [chosen], num_jitters=2
                )
                if encodings:
                    return encodings[0]
        except Exception as e:
            print(f"Upscaled face encoding failed: {e}")

        return self._orientation_fallback_encoding(rgb_image, location)

    def _orientation_fallback_encoding(self, rgb_image, location):
        """Try difficult faces at multiple orientations."""
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
                rotated, _ = self._downscale_rgb(rotated, 800)
                locations = face_recognition.face_locations(
                    rotated, number_of_times_to_upsample=1, model="hog"
                )
                if not locations:
                    continue
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
            except Exception as e:
                print(f"Orientation encoding ({angle}°) skipped: {e}")
        return best

    def _is_masked_face(self, rgb_image, location):
        """Return True when the visible face strongly suggests a face mask.

        Recognition must never mark an obviously masked student as present.
        Landmarks are preferred; a conservative skin/texture fallback catches
        common surgical/cloth masks when landmarks are incomplete.
        """
        try:
            landmarks = face_recognition.face_landmarks(
                rgb_image, [location], model="small"
            )
            if landmarks:
                lm = landmarks[0]
                has_nose = bool(lm.get("nose_bridge") or lm.get("nose_tip"))
                has_mouth = bool(lm.get("top_lip") and lm.get("bottom_lip"))
                # With a normal frontal face the nose bridge is usually visible,
                # while a mask hides the mouth/lips. Do not apply this rule to a
                # clear profile/tilted face where landmarks may naturally be absent.
                if has_nose and not has_mouth:
                    left_eye = lm.get("left_eye")
                    right_eye = lm.get("right_eye")
                    if left_eye and right_eye:
                        return True
        except Exception:
            pass

        # Conservative image heuristic for masks. Only use it when the upper
        # face has skin but the lower face has very little skin and very low
        # texture. This avoids treating a normal sideways/downward face as masked.
        try:
            top, right, bottom, left = location
            crop = rgb_image[top:bottom, left:right]
            if crop.size == 0:
                return False
            ch, cw = crop.shape[:2]
            if ch < 35 or cw < 35:
                return False

            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
            upper = hsv[int(ch * 0.22):int(ch * 0.58), int(cw * 0.18):int(cw * 0.82)]
            lower = hsv[int(ch * 0.52):int(ch * 0.92), int(cw * 0.18):int(cw * 0.82)]
            if upper.size == 0 or lower.size == 0:
                return False

            def skin_ratio(region):
                H, S, V = cv2.split(region)
                skin = (
                    (H >= 0) & (H <= 25) &
                    (S >= 25) & (S <= 190) &
                    (V >= 45)
                )
                return float(np.mean(skin))

            upper_skin = skin_ratio(upper)
            lower_skin = skin_ratio(lower)
            if upper_skin > 0.16 and lower_skin < 0.055:
                return True
        except Exception:
            pass

        return False

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

        # Encode each face independently. One difficult/sideways face must not
        # make the entire batch fail and turn every other face into Unknown.
        face_encodings_by_index = {}
        masked_faces = set()
        for idx, location in enumerate(face_locations):
            face_encodings_by_index[idx] = self._encode_face_robustly(
                classroom_image, location
            )
            if self._is_masked_face(classroom_image, location):
                masked_faces.add(idx)

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
                if face_index in masked_faces:
                    student_id = None
                    name = "Unknown"
                    confidence = 0.0
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