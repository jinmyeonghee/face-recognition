# 얼굴인식모델 개선
팀프로젝트 4인  
  
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

### 내가 진행한 부분

1) 얼굴특징 벡터 추출 (get_embedding.py)  
  - 이미지파일을 numpy배열로 변환 후 이미지(RGB)로부터 임베딩 벡터 계산
  
2) 유사도판단 코드 구현 (get_similarity.py)  
  - 얼굴이미지의 두 벡터의 거리를 측정하여 유사도를 계산하고, 유사한 이미지인지 판단.  
  - 벡터 비교방법은 코사인 거리, 유클리디안 거리, 유클리디안 L2 거리 3가지  
  - 임계값은 얼굴 인식 모델마다 다름. 베이스모델에 대한 임계값은 아래와 같음.  
  <img width="624" alt="스크린샷 2023-08-17 오후 10 57 00" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/a9f3eefc-ef10-4783-aa5a-93ec01d6368a">
  
  - 우리는 베이스모델에 학습을 하기 때문에 모델에 맞는 적당한 임계값을 찾는 것이 중요함.    
  - 통계적 접근방식으로 임계값을 찾은 다음 여러번 조정해가며 두 클래스를 구분하는 적절한 값을 찾으면 됨.      
  예시) 아래의 이미지는 두 개의 이미지쌍의 벡터거리를 나타낸 그래프(동일인 파랑, 비동일인 주황)    
  <img width="681" alt="스크린샷 2023-08-17 오후 11 04 08" src="https://github.com/jinmyeonghee/face-recognition/assets/114460314/08c1ec2c-c9ea-409f-be55-28ec73616471">
  
  - 0.3524와 0.4654사이에는 두 클래스가 모두 있음. 그 사이 적절한 임계값을 찾으면 됨.


3) Facenet512학습 및 성능 추출 (CS_AI16_Facenet512.py)  
  - 베이스모델인 Facenet512에 학습데이터를 이용하여 학습시킴. 벡터 비교방법은 코사인거리로 설정.
  - 샴네트워크에 들어갈 두 이미지를 정의하고 특징벡터 계산.
  - 거리계산 레이어와 유사성 점수 출력을 위한 시그모이드 활성화 함수를 사용하는 완전연결 레이어 추가.  
  - 최종 샴 네트워크 모델 정의.


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
