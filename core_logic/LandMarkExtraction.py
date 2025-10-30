# =====================================================================
# 3. LANDMARK INTEGRATION - MediaPipe + Dlib for Overlay Engine
# =====================================================================

import mediapipe as mp
import numpy as np
import cv2

class LandmarkExtractor:
    """
    Extract facial landmarks for overlay engine integration
    This connects your CNN classification with the overlay system
    """

    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )

        # Key landmark indices for glasses positioning
        self.key_landmarks = {
            'left_eye_outer': 33,
            'left_eye_inner': 133,
            'right_eye_inner': 362,
            'right_eye_outer': 263,
            'nose_bridge': 6,
            'nose_tip': 1,
            'left_temple': 234,
            'right_temple': 454
        }

    def extract_glasses_landmarks(self, image):
        """
        Extract specific landmarks needed for glasses overlay
        Returns coordinates for overlay engine
        """
        if self.mp_face_mesh is None:
            print("MediaPipe Face Mesh not initialized.")
            return None

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image

        results = self.mp_face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]

            h, w = image.shape[:2]

            # Extract key points for glasses positioning
            glasses_points = {}

            for name, idx in self.key_landmarks.items():
                if idx < len(landmarks.landmark):
                    lm = landmarks.landmark[idx]
                    x, y = int(lm.x * w), int(lm.y * h)
                    glasses_points[name] = (x, y)

            return glasses_points

        return None

    def calculate_glasses_transform(self, landmarks):
        """
        Calculate transformation matrix for glasses overlay
        This is what your overlay engine will use
        """
        if not landmarks:
            return None

        # Calculate glasses width and position
        left_eye_outer = landmarks.get('left_eye_outer')
        right_eye_outer = landmarks.get('right_eye_outer')

        if left_eye_outer and right_eye_outer:
            # Glasses width
            glasses_width = abs(right_eye_outer[0] - left_eye_outer[0])

            # Center position
            center_x = (left_eye_outer[0] + right_eye_outer[0]) // 2
            center_y = (left_eye_outer[1] + right_eye_outer[1]) // 2

            # Calculate rotation angle (if face is tilted)
            angle = np.arctan2(
                right_eye_outer[1] - left_eye_outer[1],
                right_eye_outer[0] - left_eye_outer[0]
            )

            return {
                'center': (center_x, center_y),
                'width': glasses_width,
                'angle': angle,
                'scale': glasses_width / 150  # Adjust based on glasses template size
            }

        return None