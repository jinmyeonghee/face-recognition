# 얼굴인식모델 개선
**프로젝트기간 : 2023.03.24 ~ 2023.04.19**

**사용 언어 : python**

**개발환경 : macOS Ventura 13.3, visual studio code**

**사용 라이브러리 : MediaPipe, OpenCV**

**사용모델 : VGG-Face, Facenet512**

***
### 프로젝트 개요
- 소개팅어플에서 회원가입 이후 본인 인증을 하기 위해 앱 사용자가 등록한 인증사진과 프로필사진을 비교하여 동일인물인지 판별할 수 있는 모델 구축

### 프로젝트 배경
- 소개팅어플을 이용한 온라인만남을 악용할 가능성이 있어 본인인증이 매우 중요 

   
## 프로젝트 내용
### 1. 프로젝트 모델 선정과 프로젝트 구조
<img width="824" alt="스크린샷 2023-08-16 오전 11 00 32" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/1dfdaa2a-a251-4f5f-817d-f62688864464">
- 두 개의 이미지를 전처리 진행한 후 벡터를 추출. 두 벡터의 거리계산을 통해 동일인인지 판단  

**프로젝트 수행절차와 개발인원들의 역할 분담**
<img width="815" alt="스크린샷 2023-08-16 오전 11 00 57" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/4d3a44aa-3e5b-41b1-8614-d16430cf17eb">

### 2. 프로젝트 수행 - 모델 구축 및 개선
<img width="940" alt="스크린샷 2023-08-16 오전 11 27 55" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/d85d224d-3ca4-46b6-b946-c0a8eabc8e6b">

### 3. 프로젝트 수행 결과
<img width="963" alt="스크린샷 2023-08-16 오전 11 28 25" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/f8634674-b44a-42be-97d5-8b19f478c712">


### 프로젝트 평가
- 최고 성능의 베이스모델을 사용했음에도 불구하고 샴네트워크를 구성하였을 때 초기화 시간도 더 오래 걸리고, 가중치크기도 커짐. 하지만 임베딩벡터를 바로 추출해서 거리비교 연산을 했을 때보다 성능이 나아지지 않았음.

  -> Verifier객체를 사용해 베이스모델을 이용하여 임베딩 벡터 유사도를 계산해주는 것이 성능이 더 좋음.

### 프로젝트 한계 및 개선방안
- 테스트에 사용한 기업데이터 자체에 얼굴이 없는 이미지들이 많이 존재하여 성능이 높이 나오지 않음
- mediapipe에서의 얼굴 이미지 오추출도 있었기때문에 학습 품질이 떨어짐
- 데이터셋 자체의 양과 tensorflow의 자체 버그로 인한 잦은 breakdown 발생
  
  -> pytorch나 onnx 등의 프레임워크를 사용한다면 버그대비 안정성이나 다양한 루트를 확보할 수 있었을 것
- Verifier2학습 과정에서 Random state비고정으로 인한 Validation셋 변동 및 섞임과 그에 따른 과적합이 유발되었을 것으로 예상됨
  
  -> train/validation set 분리 시 Random state 고정으로 과적합 최소화
- 이미지의 철저한 라벨링, 자동화를 원한다면 최소한 이미지추출 백엔드를 두개 이상 구성하여 2중 필터링 할 것

------------
------------

- 기본 트리 구조입니다.

```
    AI_16_CP2
    ├─models ─ basemodels ┬ function
    │                    (└ weights) 최초로 가중치파일을 받을 때 생성됩니다.
    │
    ├─sample_data
    └─utils ─ function
```

- 사용법
    ```
        from <AI_16_CP2까지의 경로>.face_ds_project import FaceDSProject

        # project = FaceDSProject(min_detection_confidence = 0.2, model_name = 'vggface', distance_metric = 'cosine') # 이와 같이 패러미터를 구성할 수 있습니다. 이상은 디폴트값으로, 아래와 같습니다.
        project = FaceDSProject()

        image_path1 = 'path/to/image' # 시스템 경로(str)
        image_path2 = 'http://<path/to/image>' # url주소(str)
        image_path3 = [[[255, 255, 255]
                        [255, 255, 255]
                        [255, 255, 255]
                        # ...
                        [255, 255, 255]
                        [255, 255, 255]
                        [255, 255, 255]]] # numpy.ndarray(RGB값)
        
        # 이미지에서 얼굴을 찾아서 크롭하고, 정렬하고, 패딩을 추가하고, 사이즈를 224X224로 변경해서 가져온다.
        face_list = project.get_faces(image_path1) # 통상적으로 이 메소드를 부를 일은 없습니다.
        
        verification_results = project.verify(image_path2, image_path3, threshold = 0.664)
        print(verification_results)
        # print(verification_results) 원본 이미지 6인 X 대상 이미지 6인 결과 예시
        # 성공
        {
            'result_message': '동일인이 존재합니다.',
            'result_code': 2,
            'result_list': [
                [0.8136902, 0.7838269, 0.68655384, 0.7106067, 0.5856164, 0.7276525],
                [0.7838269, 0.8136902, 0.70351696, 0.713796, 0.5963999, 0.74323696],
                [0.68655384, 0.70351696, 0.8136902, 0.7079207, 0.7185959, 0.71114814],
                [0.7106067, 0.713796, 0.7079207, 0.8136902, 0.6522614, 0.6557367],
                [0.5856164, 0.5963999, 0.7185959, 0.6522614, 0.8136902, 0.59826905],
                [0.7276525, 0.74323696, 0.71114814, 0.6557367, 0.59826905, 0.8136902]
            ]
        }
        {
            'result_message': '동일인이 존재하지 않습니다.',
            'result_code': 0,
            'result_list': [ 
                [0.2136902, 0.2838269, 0.28655384, 0.2106067, 0.2856164, 0.2276525],
                [0.2838269, 0.2136902, 0.20351696, 0.2713796, 0.2963999, 0.04323696],
                [0.08655384, 0.0351696, 0.0136902, 0.2079207, 0.0185959, 0.01114814],
                [0.0106067, 0.013796, 0.0079207, 0.0136902, 0.06522614, 0.0557367],
                [0.0856164, 0.0963999, 0.0185959, 0.0522614, 0.0136902, 0.09826905],
                [0.0276525, 0.04323696, 0.01114814, 0.0557367, 0.09826905, 0.0136902]
            ]
        }
        # 실패
        {'result_message' : '원본 이미지를 읽어올 수 없습니다.', 'result_code' : -22 }
        {'result_message' : '대상 이미지를 읽어올 수 없습니다.', 'result_code' : -21 }
        {'result_message' : '원본 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -2 }
        {'result_message' : '비교할 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -1 }

        distinction_results = project.distinguish(image_path1)
        print(distinction_results)
        # print(verification_results) 원본 이미지 6인 결과 예시
        # 성공
        {
            'result_message': '원본 이미지에서 성별을 분석했습니다.',
            'result_code': 0,
            'result_list': [
                {'gender': {'Woman': 100.0, 'Man': 0.0}, 'dominant_gender': 'Woman'},
                {'gender': {'Woman': 100.0, 'Man': 0.0}, 'dominant_gender': 'Woman'},
                {'gender': {'Woman': 99.83, 'Man': 0.17}, 'dominant_gender': 'Woman'},
                {'gender': {'Woman': 100.0, 'Man': 0.0}, 'dominant_gender': 'Woman'},
                {'gender': {'Woman': 100.0, 'Man': 0.0}, 'dominant_gender': 'Woman'},
                {'gender': {'Woman': 100.0, 'Man': 0.0}, 'dominant_gender': 'Woman'}
            ]
        }
        # 실패
        {'result_message' : '원본 이미지를 읽어올 수 없습니다.', 'result_code' : -11 }
        {'result_message' : '원본 이미지에서 얼굴이 검출되지 않았습니다.', 'result_code' : -1 }

    ```
