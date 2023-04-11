import cv2, mediapipe
from .function.align import alignment_procedure
from .function.url_to_image import url_to_np_array

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

class representer:
    def __init__(self, min_detection_confidence = 0.2):
        self.detector = mediapipe.solutions.face_detection.FaceDetection(min_detection_confidence=min_detection_confidence)

    def detect_faces(self, image_path):
        """
        얼굴 경로에서 얼굴 이미지를 불러서 numpy 배열로 만든 후 얼굴을 찾고, 얼굴 수만큼 크롭, 정렬, 패딩 추가,리사이즈해서
        리스트로 반환함
        예) 3인의 얼굴이 발견된 경우, [이미지numpy배열1, 이미지numpy배열2, 이미지numpy배열3]
        """
        image = ''
        if image_path.lower().startswith("http://") or image_path.lower().startswith("https://"):
            image = url_to_np_array(image_path)
        else:
            image = cv2.imread(image_path)
        results = self.detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        CAPR_face_list = [] # Cropped => Aligned => Padded => Resized  (CAPR)
        for idx, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cropped_face = image[y:y+h, x:x+w]

            right_eye, left_eye = get_eyes(detection, ih, iw)
            aligned_face = alignment_procedure(cropped_face, right_eye, left_eye)
            padded_face = get_padded_face(aligned_face)

            # 얼굴 이미지를 224x224로 리사이즈
            resized_face = cv2.resize(padded_face, (224, 224))
            CAPR_face_list.append(resized_face)

        return CAPR_face_list