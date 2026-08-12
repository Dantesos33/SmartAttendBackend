import face_recognition
import cv2
import numpy as np
import os
import base64
from datetime import datetime
import json

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

    def _register_encoding(self, image_path, student_id, name, persist_only=False):
        try:
            image = face_recognition.load_image_file(image_path)
            locations = face_recognition.face_locations(image)
            if not locations:
                return False, "No face detected in image.", None
            if len(locations) > 1:
                return False, "Multiple faces detected — please use a photo with only one person.", None

            encodings = face_recognition.face_encodings(
                image,
                [locations[0]],
                num_jitters=2,
            )
            if not encodings:
                encodings = face_recognition.face_encodings(image, [locations[0]], num_jitters=1)
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
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                return False, "No face detected. Please upload a clear photo of your face."
            if len(encodings) > 1:
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

    def _encode_bgr_jpeg_base64(self, bgr_image, quality=82):
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

    def _flip_locations(self, locations, width):
        flipped = []
        for top, right, bottom, left in locations:
            flipped.append((top, width - left, bottom, width - right))
        return flipped

    def _detect_face_locations(self, rgb_image):
        """Multi-pass detection: padded edges + multiple scales + upsampling."""
        h, w = rgb_image.shape[:2]
        pad_y = max(20, int(h * 0.08))
        pad_x = max(20, int(w * 0.08))
        padded = cv2.copyMakeBorder(
            rgb_image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_REPLICATE
        )

        collected = []
        for source, off_y, off_x in ((padded, pad_y, pad_x), (rgb_image, 0, 0)):
            for scale in (1.0, 1.25, 1.5):
                if scale == 1.0:
                    img = source
                else:
                    sh, sw = source.shape[:2]
                    img = cv2.resize(source, (int(sw * scale), int(sh * scale)))
                for upsample in (1, 2, 3):
                    batch = face_recognition.face_locations(
                        img,
                        number_of_times_to_upsample=upsample,
                    )
                    batch = self._scale_locations(batch, scale)
                    batch = self._offset_locations(batch, off_y, off_x)
                    collected.extend(batch)

        flipped = np.fliplr(rgb_image)
        for upsample in (1, 2):
            batch = face_recognition.face_locations(
                flipped,
                number_of_times_to_upsample=upsample,
            )
            collected.extend(self._flip_locations(batch, w))

        merged = self._merge_face_locations(collected)
        print(f"Detected {len(merged)} face(s) after multi-pass detection.")
        return merged

    def _encode_face_at_location(self, rgb_image, location):
        for jitters in (2, 1):
            encodings = face_recognition.face_encodings(
                rgb_image,
                [location],
                num_jitters=jitters,
            )
            if encodings:
                return encodings[0]
        return None

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
        # Reload from disk so newly registered profile photos are included.
        self.load_known_students_from_dir()
        if db_encoding_rows:
            self.load_db_encodings(db_encoding_rows)

        if not self.known_face_encodings:
            return None, "No student faces registered yet."

        allowed_set = set(allowed_student_ids) if allowed_student_ids else None

        enrolled_with_faces = None
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

        present_student_ids = []
        unknown_faces = 0
        face_details = []

        for face_index, location in enumerate(face_locations):
            top, right, bottom, left = location
            face_encoding = self._encode_face_at_location(classroom_image, location)

            student_id = None
            name = "Unknown"
            confidence = 0.0

            if face_encoding is None:
                unknown_faces += 1
            elif allowed_set is not None and not enrolled_with_faces:
                unknown_faces += 1
            else:
                student_id, name, confidence = self._match_face(
                    face_encoding,
                    tolerance=tolerance,
                    allowed_set=allowed_set,
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
            "face_details": face_details,
        }

        os.makedirs("output", exist_ok=True)
        annotated_path = f"output/annotated_classroom_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(annotated_path, classroom_image_cv)
        attendance_data["annotated_image_path"] = annotated_path
        attendance_data["annotated_image_base64"] = self._encode_bgr_jpeg_base64(
            classroom_image_cv
        )

        return attendance_data, "Recognition complete!"
