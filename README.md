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