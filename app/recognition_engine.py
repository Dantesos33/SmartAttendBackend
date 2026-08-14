import face_recognition
import cv2
import numpy as np
import os
import base64
import gc
from datetime import datetime
import json

MIN_CONFIDENCE = 0.45
MAX_DETECT_EDGE = 1600
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
            cv2.imwrite(safe_path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 97, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
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

    def _encode_rgb_crop_base64(self, rgb_image, top, right, bottom, left, quality=94):
        """Return a high-quality, padded profile crop suitable for student enrollment.

        Detection boxes around distant students can be only a few dozen pixels wide.
        Sending that tiny rectangle directly to the app makes the resulting profile
        photo look badly pixelated. We therefore crop from the ORIGINAL classroom
        image, add generous context around the face, and upscale with Lanczos before
        JPEG encoding at high quality. This does not invent facial detail, but avoids
        throwing away the available source pixels and JPEG quality.
        """
        try:
            h, w = rgb_image.shape[:2]
            top, right, bottom, left = map(int, (top, right, bottom, left))
            top = max(0, min(h - 1, top))
            left = max(0, min(w - 1, left))
            bottom = max(top + 1, min(h, bottom))
            right = max(left + 1, min(w, right))

            face_w = right - left
            face_h = bottom - top
            if face_w < 16 or face_h < 16:
                return None

            # Include head/shoulders instead of returning a tiny face-only box.
            # Use a square crop so it works cleanly as a profile/avatar image.
            crop_size = int(round(max(face_w, face_h) * 2.4))
            crop_size = max(crop_size, 192)
            crop_size = min(crop_size, max(h, w))

            cx = (left + right) / 2.0
            cy = top + face_h * 0.62
            x0 = int(round(cx - crop_size / 2.0))
            y0 = int(round(cy - crop_size / 2.0))
            x1 = x0 + crop_size
            y1 = y0 + crop_size

            # Shift the square back inside the original image rather than padding
            # with artificial pixels.
            if x0 < 0:
                x1 -= x0
                x0 = 0
            if y0 < 0:
                y1 -= y0
                y0 = 0
            if x1 > w:
                x0 -= x1 - w
                x1 = w
            if y1 > h:
                y0 -= y1 - h
                y1 = h
            x0, y0 = max(0, x0), max(0, y0)

            crop = rgb_image[y0:y1, x0:x1]
            if crop.size == 0:
                return None

            crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

            # Always provide enough pixels for a usable profile photo.
            target = 512
            if crop_bgr.shape[0] != target or crop_bgr.shape[1] != target:
                crop_bgr = cv2.resize(
                    crop_bgr,
                    (target, target),
                    interpolation=cv2.INTER_LANCZOS4,
                )

            # Very light sharpening compensates for the interpolation without
            # creating the harsh halos produced by aggressive sharpening.
            blurred = cv2.GaussianBlur(crop_bgr, (0, 0), 0.7)
            crop_bgr = cv2.addWeighted(crop_bgr, 1.08, blurred, -0.08, 0)

            ok, buffer = cv2.imencode(
                ".jpg", crop_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, quality,
                 cv2.IMWRITE_JPEG_OPTIMIZE, 1],
            )
            if not ok:
                return None
            return base64.b64encode(buffer).decode("utf-8")
        except Exception as e:
            print(f"Profile crop encoding failed: {e}")
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

    @staticmethod
    def _scale_locations(locations, scale, width, height):
        inv = 1.0 / scale
        return [
            (
                max(0, int(round(top * inv))),
                min(width, int(round(right * inv))),
                min(height, int(round(bottom * inv))),
                max(0, int(round(left * inv))),
            )
            for top, right, bottom, left in locations
        ]

    def _detect_face_locations(self, rgb_image):
        """Memory-safe high-recall detector.

        The normal HOG pass runs first. If it misses faces, detection is retried
        on small overlapping tiles with upsample=2. Tiling is intentional: a
        full-image upsample=3 pass can consume hundreds of MB on Railway for a
        single classroom photo. Tiles give the detector more pixels around small
        faces without creating one enormous dlib image pyramid.
        """
        h, w = rgb_image.shape[:2]
        detections = []

        def add(box, iou=0.30):
            top, right, bottom, left = [int(v) for v in box]
            top = max(0, min(h - 1, top))
            left = max(0, min(w - 1, left))
            bottom = max(top + 1, min(h, bottom))
            right = max(left + 1, min(w, right))
            if bottom - top < 20 or right - left < 20:
                return
            candidate = (top, right, bottom, left)
            if not any(self._box_iou(candidate, x) >= iou for x in detections):
                detections.append(candidate)

        detect_img, scale = self._downscale_rgb(rgb_image, MAX_DETECT_EDGE)
        try:
            locations = face_recognition.face_locations(
                detect_img, number_of_times_to_upsample=1, model="hog"
            )
            for box in self._scale_locations(locations, scale, w, h):
                add(box)
        except Exception as e:
            print(f"Primary face detection failed: {e}")
        finally:
            del detect_img

        # Small-face recovery. Use overlapping tiles instead of upsample=3 on the
        # entire image. At most one tile is held by dlib at a time.
        if len(detections) < 4:
            tile_max = 900
            overlap = 0.28
            step = max(1, int(tile_max * (1.0 - overlap)))
            ys = list(range(0, max(1, h - tile_max + 1), step))
            xs = list(range(0, max(1, w - tile_max + 1), step))
            if not ys or ys[-1] != max(0, h - tile_max):
                ys.append(max(0, h - tile_max))
            if not xs or xs[-1] != max(0, w - tile_max):
                xs.append(max(0, w - tile_max))

            for y0 in ys:
                for x0 in xs:
                    y1 = min(h, y0 + tile_max)
                    x1 = min(w, x0 + tile_max)
                    tile = np.ascontiguousarray(rgb_image[y0:y1, x0:x1])
                    try:
                        locations = face_recognition.face_locations(
                            tile, number_of_times_to_upsample=2, model="hog"
                        )
                        for top, right, bottom, left in locations:
                            add((top + y0, right + x0, bottom + y0, left + x0))
                    except Exception as e:
                        print(f"Tile face detection failed at ({x0},{y0}): {e}")
                    finally:
                        del tile
                    if len(detections) >= 4:
                        break
                if len(detections) >= 4:
                    break

        # Profile fallback uses a downscaled grayscale copy, not the full RGB
        # image. This is substantially cheaper than rotating the whole image.
        if len(detections) < 4:
            profile_img, profile_scale = self._downscale_rgb(rgb_image, 1400)
            gray = None
            try:
                gray = cv2.cvtColor(profile_img, cv2.COLOR_RGB2GRAY)
                cascade = cv2.CascadeClassifier(
                    os.path.join(cv2.data.haarcascades, "haarcascade_profileface.xml")
                )
                if not cascade.empty():
                    boxes = cascade.detectMultiScale(
                        gray, scaleFactor=1.08, minNeighbors=5, minSize=(24, 24)
                    )
                    for x, y, bw, bh in boxes:
                        add(tuple(self._scale_locations(
                            [(y, x + bw, y + bh, x)], profile_scale, w, h
                        )[0]), iou=0.25)
            except Exception as e:
                print(f"Profile fallback failed: {e}")
            finally:
                del gray
                del profile_img

        detections.sort(key=lambda b: (b[0], b[3]))
        print(f"Detected {len(detections)} face(s) in image.")
        return detections

    def _encode_face(self, rgb_image, location):
        """Encode one detected face independently so one hard face cannot break all faces."""
        try:
            encodings = face_recognition.face_encodings(
                rgb_image, known_face_locations=[location], num_jitters=1
            )
            if encodings:
                return encodings[0]
        except Exception as e:
            print(f"Face encoding failed for one face: {e}")
        return None

    def _is_masked_face(self, rgb_image, location):
        """Conservative mask check: missing mouth with visible eyes/nose means Unknown."""
        try:
            landmarks = face_recognition.face_landmarks(
                rgb_image, [location], model="small"
            )
            if not landmarks:
                return False
            lm = landmarks[0]
            has_nose = bool(lm.get("nose_bridge") or lm.get("nose_tip"))
            has_mouth = bool(lm.get("top_lip") and lm.get("bottom_lip"))
            has_eyes = bool(lm.get("left_eye") and lm.get("right_eye"))
            return has_eyes and has_nose and not has_mouth
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

        # Encode each face independently. A difficult/sideways face must not
        # cause every other face in the photo to become Unknown.
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
                # A masked face is never considered a recognized attendance
                # match, even if the visible upper-face embedding is close.
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