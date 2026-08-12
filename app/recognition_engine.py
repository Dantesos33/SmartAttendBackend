import face_recognition
import cv2
import numpy as np
import os
import base64
import gc
from datetime import datetime
import json

MIN_CONFIDENCE = 0.48
MAX_DETECT_EDGE = 1280
MAX_ANNOTATED_EDGE = 1200

class ClassroomAttendanceSystem:
    """
    Keyed by real student_id (the database's users.id) rather than a typed
    name string. This is what lets a recognition result map directly back to
    a real enrolled student — there's no more "reconcile this name string
    against the roster" step, which used to be the actual gap in the system.

    known_face_ids[i] / known_face_encodings[i] / known_face_names[i] are
    parallel lists — names are kept only for display/logging, never used as
    the actual key.
    """

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
                with open(path, "r") as f:
                    return json.load(f)
            except Exception:
                return default
        return default

    def _save_json(self, path, data):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)

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
        """Add or replace one student's embedding in memory."""
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
        """Merge face embeddings persisted in the database."""
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
        """Load every registered face from disk. Files are named
        {student_id}.jpg — metadata.json (keyed by student_id as a string,
        since JSON object keys are always strings) holds the display name."""
        print("Loading known students...")
        self.known_face_encodings = []
        self.known_face_ids = []
        self.known_face_names = []

        for filename in os.listdir(self.known_students_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filepath = os.path.join(self.known_students_dir, filename)
                stem = os.path.splitext(filename)[0]
                if not stem.isdigit():
                    # Skip any leftover legacy name-keyed files from before
                    # this rework — they have no real student_id to attach to.
                    continue
                student_id = int(stem)
                name = self.metadata.get(str(student_id), {}).get("name", f"Student {student_id}")
                self._register_encoding(filepath, student_id, name)
        print(f"Loaded {len(self.known_face_ids)} known students.")

    def _register_encoding(self, image_path, student_id, name):
        try:
            image = face_recognition.load_image_file(image_path)
            image, _ = self._downscale_rgb(image, MAX_DETECT_EDGE)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
            if not locations:
                return False, "No face detected in image.", None
            if len(locations) > 1:
                return False, "Multiple faces detected — please use a photo with only one person.", None

            encodings = face_recognition.face_encodings(
                image,
                [locations[0]],
                num_jitters=1,
            )
            if not encodings:
                return False, "Could not encode face from image.", None

            if student_id in self.known_face_ids:
                idx = self.known_face_ids.index(student_id)
                self.known_face_encodings[idx] = encodings[0]
                self.known_face_names[idx] = name
            else:
                self.known_face_encodings.append(encodings[0])
                self.known_face_ids.append(student_id)
                self.known_face_names.append(name)
            return True, f"Successfully registered face for: {name}", encodings[0]
        except Exception as e:
            return False, f"Error processing image: {str(e)}", None

    def verify_face_quality(self, image_path):
        """Check-only, no registration — used to validate a photo upload
        before accepting it as a profile photo (exactly one clear face)."""
        try:
            image = face_recognition.load_image_file(image_path)
            image, _ = self._downscale_rgb(image, MAX_DETECT_EDGE)
            locations = face_recognition.face_locations(image, number_of_times_to_upsample=1)
            if len(locations) == 0:
                return False, "No face detected. Please upload a clear photo of your face."
            if len(locations) > 1:
                return False, "Multiple faces detected. Please upload a photo with only yourself in frame."
            return True, "Face detected clearly."
        except Exception as e:
            return False, f"Couldn't process this image: {str(e)}"

    def register_student_face(self, image_path, student_id, name, roll=None):
        """Registers (or re-registers) a student's face, keyed by their real
        database student_id. Called from the profile-photo-upload endpoint."""
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

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=85):
        h, w = rgb_image.shape[:2]
        top = max(0, top)
        left = max(0, left)
        bottom = min(h, bottom)
        right = min(w, right)
        if bottom <= top or right <= left:
            return None
        crop = rgb_image[top:bottom, left:right]
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        return self._encode_bgr_jpeg_base64(crop_bgr, quality=quality)

    @staticmethod
    def _box_area(top, right, bottom, left):
        return max(0, bottom - top) * max(0, right - left)

    @staticmethod
    def _boxes_overlap(a, b):
        a_top, a_right, a_bottom, a_left = a
        b_top, b_right, b_bottom, b_left = b
        inter_top = max(a_top, b_top)
        inter_left = max(a_left, b_left)
        inter_bottom = min(a_bottom, b_bottom)
        inter_right = min(a_right, b_right)
        inter_area = ClassroomAttendanceSystem._box_area(
            inter_top, inter_right, inter_bottom, inter_left
        )
        if inter_area <= 0:
            return False
        a_area = ClassroomAttendanceSystem._box_area(a_top, a_right, a_bottom, a_left)
        b_area = ClassroomAttendanceSystem._box_area(b_top, b_right, b_bottom, b_left)
        union = a_area + b_area - inter_area
        return union > 0 and (inter_area / union) >= 0.35

    def _merge_face_locations(self, locations):
        """Deduplicate overlapping detections, keeping the largest box."""
        unique = []
        for loc in sorted(locations, key=lambda box: self._box_area(*box), reverse=True):
            if not any(self._boxes_overlap(loc, kept) for kept in unique):
                unique.append(loc)
        return unique

    def _scale_locations(self, locations, scale):
        if scale == 1.0:
            return locations
        inv = 1.0 / scale
        return [
            (int(top * inv), int(right * inv), int(bottom * inv), int(left * inv))
            for top, right, bottom, left in locations
        ]

    def _offset_locations(self, locations, offset_y, offset_x):
        return [
            (top - offset_y, right - offset_x, bottom - offset_y, left - offset_x)
            for top, right, bottom, left in locations
        ]

    @staticmethod
    def _downscale_rgb(rgb_image, max_edge):
        h, w = rgb_image.shape[:2]
        longest = max(h, w)
        if longest <= max_edge:
            return rgb_image, 1.0
        scale = max_edge / longest
        resized = cv2.resize(
            rgb_image,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def prepare_known_faces(self, db_encoding_rows=None, allowed_student_ids=None):
        """Load only the embeddings needed for this recognition request."""
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
        """Memory-safe detection: downscale first, add extra passes only if needed."""
        detect_img, det_scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        inv_scale = 1.0 / det_scale
        collected = []

        def add_pass(img, offset_y=0, offset_x=0, upsample=1):
            batch = face_recognition.face_locations(
                img,
                number_of_times_to_upsample=upsample,
            )
            if offset_y or offset_x:
                batch = self._offset_locations(batch, offset_y, offset_x)
            collected.extend(self._scale_locations(batch, inv_scale))

        add_pass(detect_img, upsample=1)
        merged = self._merge_face_locations(collected)

        if len(merged) < 4:
            h, w = detect_img.shape[:2]
            pad = max(12, int(min(h, w) * 0.05))
            padded = cv2.copyMakeBorder(
                detect_img, pad, pad, pad, pad, cv2.BORDER_REPLICATE
            )
            add_pass(padded, offset_y=pad, offset_x=pad, upsample=1)
            del padded
            merged = self._merge_face_locations(collected)

        if len(merged) < 4:
            add_pass(detect_img, upsample=2)
            merged = self._merge_face_locations(collected)

        del detect_img
        gc.collect()
        print(f"Detected {len(merged)} face(s) after memory-safe detection.")
        return merged

    def _encode_face_at_location(self, rgb_image, location):
        work_image, scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        if scale != 1.0:
            top, right, bottom, left = location
            location = (
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale),
            )
        encodings = face_recognition.face_encodings(
            work_image,
            [location],
            num_jitters=1,
        )
        if encodings:
            return encodings[0]
        return None

    def _assign_faces_to_students(
        self,
        face_encodings_by_index: dict[int, np.ndarray | None],
        tolerance: float,
        allowed_set: set | None,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict[int, tuple[int | None, str, float]]:
        """One enrolled student can match at most one face; weak matches become Unknown."""
        candidates: list[tuple[int, int, str, float, float]] = []

        for face_index, encoding in face_encodings_by_index.items():
            if encoding is None:
                continue
            distances = face_recognition.face_distance(self.known_face_encodings, encoding)
            for idx in np.argsort(distances):
                distance = float(distances[idx])
                if distance > tolerance:
                    break
                student_id = self.known_face_ids[idx]
                if allowed_set is not None and student_id not in allowed_set:
                    continue
                confidence = 1 - distance
                if confidence < min_confidence:
                    continue
                candidates.append(
                    (
                        face_index,
                        student_id,
                        self.known_face_names[idx],
                        confidence,
                        distance,
                    )
                )

        candidates.sort(key=lambda item: item[4])

        assigned_faces: set[int] = set()
        assigned_students: set[int] = set()
        assignments: dict[int, tuple[int | None, str, float]] = {}

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

    def _match_face(self, face_encoding, tolerance, allowed_set=None):
        """Pick the closest known face within tolerance; when section-scoped,
        skip matches for students outside the allowed enrollment set."""
        if not self.known_face_encodings:
            return None, "Unknown", 0.0

        distances = face_recognition.face_distance(
            self.known_face_encodings, face_encoding
        )
        for idx in np.argsort(distances):
            distance = float(distances[idx])
            if distance > tolerance:
                break
            student_id = self.known_face_ids[idx]
            if allowed_set is not None and student_id not in allowed_set:
                continue
            name = self.known_face_names[idx]
            return student_id, name, 1 - distance
        return None, "Unknown", 0.0

    def recognize_classroom(
        self,
        classroom_image_path,
        tolerance=0.65,
        allowed_student_ids=None,
        db_encoding_rows=None,
    ):
        """Recognize students in a classroom image. Returns student_id (real
        database id) for every match, not just a name string.

        When allowed_student_ids is provided, only students enrolled in the
        target section can be matched — others appear as unknown faces."""
        # Load only section-scoped embeddings — avoid reloading every student photo.
        self.prepare_known_faces(
            db_encoding_rows=db_encoding_rows,
            allowed_student_ids=allowed_student_ids,
        )

        has_registered_faces = bool(self.known_face_encodings)
        if not has_registered_faces:
            print(
                "No registered face embeddings — running detection-only mode "
                "(faces will appear as Unknown until students upload profile photos)."
            )

        allowed_set = set(allowed_student_ids) if allowed_student_ids else None

        enrolled_with_faces = []
        if allowed_set is not None:
            enrolled_with_faces = [
                sid for sid in allowed_set if sid in self.known_face_ids
            ]
            print(
                f"Section scope: {len(allowed_set)} enrolled, "
                f"{len(enrolled_with_faces)} with registered face photos."
            )

        print("Analyzing classroom image...")

        classroom_image = face_recognition.load_image_file(classroom_image_path)
        classroom_image_cv = cv2.cvtColor(classroom_image, cv2.COLOR_RGB2BGR)

        face_locations = self._detect_face_locations(classroom_image)

        face_encodings_by_index: dict[int, np.ndarray | None] = {}
        for face_index, location in enumerate(face_locations):
            face_encodings_by_index[face_index] = self._encode_face_at_location(
                classroom_image, location
            )

        assignments: dict[int, tuple[int | None, str, float]] = {}
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

            if face_encoding is None:
                unknown_faces += 1
            elif not has_registered_faces:
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

            color = (0, 255, 0) if student_id is not None else (0, 0, 255)
            cv2.rectangle(classroom_image_cv, (left, top), (right, bottom), color, 2)
            label = f"{name} ({confidence:.0%})" if student_id is not None else "Unknown"
            cv2.rectangle(
                classroom_image_cv, (left, bottom - 35), (right, bottom), color, cv2.FILLED
            )
            cv2.putText(
                classroom_image_cv,
                label,
                (left + 6, bottom - 6),
                cv2.FONT_HERSHEY_DUPLEX,
                0.5,
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
                else "No enrolled students have profile photos registered on the server. "
                "Faces were detected but all are marked Unknown — ask each student to "
                "upload their profile photo, then try again."
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
