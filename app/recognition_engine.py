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
            encodings = face_recognition.face_encodings(image)
            if len(encodings) == 0:
                return False, "No face detected in image."
            if len(encodings) > 1:
                return False, "Multiple faces detected — please use a photo with only one person."

            if student_id in self.known_face_ids:
                idx = self.known_face_ids.index(student_id)
                self.known_face_encodings[idx] = encodings[0]
                self.known_face_names[idx] = name
            else:
                self.known_face_encodings.append(encodings[0])
                self.known_face_ids.append(student_id)
                self.known_face_names.append(name)
            return True, f"Successfully registered face for: {name}"
        except Exception as e:
            return False, f"Error processing image: {str(e)}"

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

        success, message = self._register_encoding(safe_path, student_id, name)
        if success:
            self.metadata[str(student_id)] = {"name": name, "roll": roll}
            self._save_json(self.metadata_path, self.metadata)
        elif os.path.exists(safe_path) and safe_path != image_path:
            os.remove(safe_path)
        return success, message

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

    def recognize_classroom(
        self,
        classroom_image_path,
        tolerance=0.45,
        allowed_student_ids=None,
        min_confidence=0.55,
    ):
        """Recognize students in a classroom image. Returns student_id (real
        database id) for every match, not just a name string.

        When allowed_student_ids is provided, only students enrolled in the
        target section can be matched — others appear as unknown faces."""
        if not self.known_face_encodings:
            return None, "No student faces registered yet."

        allowed_set = set(allowed_student_ids) if allowed_student_ids else None

        candidate_indices = list(range(len(self.known_face_ids)))
        if allowed_set is not None:
            candidate_indices = [
                i for i, sid in enumerate(self.known_face_ids) if sid in allowed_set
            ]

        print("Analyzing classroom image...")

        classroom_image = face_recognition.load_image_file(classroom_image_path)
        classroom_image_cv = cv2.cvtColor(classroom_image, cv2.COLOR_RGB2BGR)

        # Upsample improves detection of smaller / distant faces in group photos.
        face_locations = face_recognition.face_locations(
            classroom_image,
            number_of_times_to_upsample=2,
        )
        face_encodings = face_recognition.face_encodings(
            classroom_image,
            face_locations,
            num_jitters=1,
        )

        if len(face_encodings) < len(face_locations):
            print(
                f"Warning: encoded {len(face_encodings)} of {len(face_locations)} detected faces"
            )

        present_student_ids = []
        unknown_faces = 0
        face_details = []

        subset_encodings = [self.known_face_encodings[i] for i in candidate_indices]
        subset_ids = [self.known_face_ids[i] for i in candidate_indices]
        subset_names = [self.known_face_names[i] for i in candidate_indices]

        for face_index, face_encoding in enumerate(face_encodings):
            if face_index >= len(face_locations):
                break
            top, right, bottom, left = face_locations[face_index]

            student_id = None
            name = "Unknown"
            confidence = 0.0

            if subset_encodings:
                matches = face_recognition.compare_faces(
                    subset_encodings, face_encoding, tolerance=tolerance
                )
                face_distances = face_recognition.face_distance(
                    subset_encodings, face_encoding
                )
                best_match_index = int(np.argmin(face_distances))
                best_confidence = 1 - face_distances[best_match_index]
                if matches[best_match_index] and best_confidence >= min_confidence:
                    student_id = subset_ids[best_match_index]
                    name = subset_names[best_match_index]
                    confidence = float(best_confidence)
                    if student_id not in present_student_ids:
                        present_student_ids.append(student_id)
                else:
                    unknown_faces += 1
            else:
                unknown_faces += 1

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
        else:
            absent_student_ids = [
                sid for sid in self.known_face_ids if sid not in present_student_ids
            ]

        attendance_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M:%S"),
            "total_registered": len(subset_ids) if allowed_set is not None else len(self.known_face_ids),
            "present_student_ids": present_student_ids,
            "absent_student_ids": absent_student_ids,
            "present_count": len(present_student_ids),
            "absent_count": len(absent_student_ids),
            "unknown_faces": unknown_faces,
            "faces_detected": len(face_encodings),
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
