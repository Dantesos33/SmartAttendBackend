import face_recognition
import cv2
import numpy as np
import os
import base64
import gc
from datetime import datetime
import json

MIN_CONFIDENCE = 0.0  # Do not silently tighten the API tolerance.
MAX_DETECT_EDGE = 2000
MAX_ANNOTATED_EDGE = 1200


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
        inter_top = max(at, bt)
        inter_left = max(al, bl)
        inter_bottom = min(ab, bb)
        inter_right = min(ar, br)
        iw = max(0, inter_right - inter_left)
        ih = max(0, inter_bottom - inter_top)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0, ar - al) * max(0, ab - at)
        area_b = max(0, br - bl) * max(0, bb - bt)
        return inter / float(area_a + area_b - inter + 1e-6)

    def _merge_locations(self, locations, iou_threshold=0.30):
        """Merge overlapping detections from HOG/Haar/tile passes."""
        h, w = None, None
        unique = []
        for loc in sorted(locations, key=lambda b: (b[2]-b[0])*(b[1]-b[3]), reverse=True):
            top, right, bottom, left = map(int, loc)
            if bottom <= top or right <= left:
                continue
            if bottom - top < 18 or right - left < 18:
                continue
            if not any(self._box_iou((top, right, bottom, left), kept) >= iou_threshold for kept in unique):
                unique.append((top, right, bottom, left))
        return unique

    def _detect_face_locations(self, rgb_image):
        """High-recall, Railway-safe detection.

        HOG remains the primary detector. Haar frontal/profile cascades are used
        as a lightweight fallback for small, masked, sideways, or tilted faces.
        If those still miss a face, small overlapping tiles are processed one at
        a time with HOG upsample=2; this avoids the RAM spike caused by a full
        image upsample=3 pass.
        """
        h, w = rgb_image.shape[:2]
        detect_img, scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        inv = 1.0 / scale
        detections = []

        def add_full(box):
            top, right, bottom, left = box
            detections.append((
                max(0, int(round(top * inv))),
                min(w, int(round(right * inv))),
                min(h, int(round(bottom * inv))),
                max(0, int(round(left * inv))),
            ))

        # 1) Reliable baseline.
        try:
            for box in face_recognition.face_locations(
                detect_img, number_of_times_to_upsample=1, model="hog"
            ):
                add_full(box)
        except Exception as e:
            print(f"Primary HOG detection failed: {e}")

        merged = self._merge_locations(detections)

        # 2) Lightweight Haar fallback. On the supplied 1200x900 image this
        # detects the far-right fourth face that HOG misses, and it also helps
        # with masks because the eyes/upper face remain visible.
        try:
            gray = cv2.cvtColor(detect_img, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)
            cascades = (
                "haarcascade_frontalface_default.xml",
                "haarcascade_frontalface_alt2.xml",
                "haarcascade_profileface.xml",
            )
            for filename in cascades:
                cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, filename))
                if cascade.empty():
                    continue
                boxes = cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.08,
                    minNeighbors=4,
                    minSize=(24, 24),
                )
                max_dim = int(min(gray.shape[:2]) * 0.22)
                for x, y, bw, bh in boxes:
                    # Haar can produce large false positives on boxes, bags,
                    # and clothing. Keep only face-sized candidates in the
                    # upper part of the group image; HOG remains authoritative
                    # for large/close faces.
                    if max(bw, bh) > max_dim:
                        continue
                    if (y + bh / 2.0) > gray.shape[0] * 0.80:
                        continue
                    add_full((y, x + bw, y + bh, x))
            merged = self._merge_locations(detections)
        except Exception as e:
            print(f"Haar fallback failed: {e}")

        # 3) If we still have fewer than four faces, use sequential overlapping
        # tiles. Only one tile exists at a time, keeping Railway memory low.
        if len(merged) < 4:
            dh, dw = detect_img.shape[:2]
            tile_h = max(320, int(dh * 0.68))
            tile_w = max(320, int(dw * 0.68))
            step_y = max(160, int(tile_h * 0.58))
            step_x = max(160, int(tile_w * 0.58))

            for y0 in range(0, max(1, dh - tile_h + 1), step_y):
                for x0 in range(0, max(1, dw - tile_w + 1), step_x):
                    y1 = min(dh, y0 + tile_h)
                    x1 = min(dw, x0 + tile_w)
                    tile = np.ascontiguousarray(detect_img[y0:y1, x0:x1])
                    try:
                        boxes = face_recognition.face_locations(
                            tile, number_of_times_to_upsample=2, model="hog"
                        )
                        for top, right, bottom, left in boxes:
                            add_full((top + y0, right + x0, bottom + y0, left + x0))
                    except Exception as e:
                        print(f"Tile HOG detection failed: {e}")
                    del tile
                    merged = self._merge_locations(detections)
                    if len(merged) >= 4:
                        break
                if len(merged) >= 4:
                    break

        del detect_img
        gc.collect()

        merged = self._merge_locations(detections)
        merged.sort(key=lambda b: (b[0], b[3]))
        print(f"Detected {len(merged)} face(s) in image.")
        return merged

    def _encode_face(self, rgb_image, location):
        """Encode one face at a time so a difficult face cannot break the batch."""
        try:
            encodings = face_recognition.face_encodings(
                rgb_image, known_face_locations=[location], num_jitters=1
            )
            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"Face encoding failed: {e}")
        return None

    def _is_masked_face(self, rgb_image, location):
        """Conservative mask check: if eyes are detected but the mouth is not,
        force the face to Unknown. Detection itself is never discarded."""
        try:
            top, right, bottom, left = location
            face = rgb_image[max(0, top):min(rgb_image.shape[0], bottom),
                             max(0, left):min(rgb_image.shape[1], right)]
            if face.size == 0 or min(face.shape[:2]) < 40:
                return False
            gray = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
            eye = cv2.CascadeClassifier(
                os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml")
            )
            if eye.empty():
                return False
            eyes = eye.detectMultiScale(gray, 1.08, 4, minSize=(10, 10))
            if len(eyes) < 2:
                return False
            # Prefer the actual face landmarks when available. A normal face
            # has a mouth contour; a covered lower face often does not.
            landmarks = face_recognition.face_landmarks(rgb_image, [location], model="small")
            if landmarks:
                lm = landmarks[0]
                if lm.get("top_lip") and lm.get("bottom_lip"):
                    return False
                if lm.get("left_eye") and lm.get("right_eye"):
                    return True
            return False
        except Exception:
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
                if min_confidence > 0.0 and confidence < min_confidence:
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

        # Encode each face independently. This prevents one small/masked/sideways
        # face from causing every other face in the image to become Unknown.
        face_encodings_by_index = {}
        masked_faces = set()
        for idx, location in enumerate(face_locations):
            face_encodings_by_index[idx] = self._encode_face(classroom_image, location)
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
                    student_id, name, confidence = None, "Unknown", 0.0
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