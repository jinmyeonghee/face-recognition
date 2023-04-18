import cv2, mediapipe, os
from .function.align import alignment_procedure
from .function.generals import find_target_size


def get_eyes(detection, ih, iw):
    landmarks = detection.location_data.relative_keypoints
    right_eye = (int(landmarks[0].x * iw), int(landmarks[0].y * ih))
    left_eye = (int(landmarks[1].x * iw), int(landmarks[1].y * ih))
    return right_eye, left_eye

def get_padded_face(aligned_face):
    face_height, face_width, _ = aligned_face.shape
    if face_width > face_height:
        padding = int((face_width - face_height) / 2)
        return cv2.copyMakeBorder(aligned_face, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        padding = int((face_height - face_width) / 2)
        return cv2.copyMakeBorder(aligned_face, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))

class FacePreparer:
    def __init__(self, min_detection_confidence = 0.2):
        self.detector = mediapipe.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect_faces(self, image, model_name='vggface', align=True):
        """
        RGB이미지 형식의 np.ndarray로 받은 후 얼굴을 찾고, 얼굴 수만큼 크롭, 정렬, 패딩 추가,리사이즈해서
        리스트로 반환함. 각 요소는 (0 ~ 255)값
        예) 3인의 얼굴이 발견된 경우, [이미지np.ndarray1, 이미지np.ndarray2, 이미지np.ndarray3]
        """
        results = self.detector.process(image)  
        target_size = find_target_size(model_name.lower())

        CAPR_face_list = [] # Cropped => Aligned => Padded => Resized  (CAPR)
        if results.detections:
            for idx, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                # Crop
                x, y, w, h = int(max(bboxC.xmin * iw, 0)), int(max(bboxC.ymin * ih, 0)), int(bboxC.width * iw), int(bboxC.height * ih)  # 음수값이 나와 이미지 크기를 벗어나지 않도록 함
                cropped_face = image[y:y+h, x:x+w]
                # Align  
                if align:
                    right_eye, left_eye = get_eyes(detection, ih, iw)
                    aligned_face = alignment_procedure(cropped_face, right_eye, left_eye)
                else:
                    aligned_face = cropped_face
                # Padding     
                padded_face = get_padded_face(aligned_face)

                # 얼굴 이미지를 target_size로 리사이즈
                resized_face = cv2.resize(padded_face, target_size)
                CAPR_face_list.append(resized_face)

        return CAPR_face_list
