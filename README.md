- 기본 트리 구조입니다.

```
    AI_16_CP2
    ├─models ─ basemodels (─ weights) 최초로 가중치파일을 받을 때 생성됩니다.
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
        face_img_list = project.get_faces(image_path1)
        
        verification_results = project.verify(image_path2, image_path3)
        print(verification_results)
        # print(verification_results) 결과 예시
        [{'1번째 얼굴': [ ('임계값: 0.4, 두 얼굴의 유사도: 0.2',   True),
            ('임계값: 0.4, 두 얼굴의 유사도: 0.64', False) ]},
        {'2번째 얼굴': [{'임계값': 0.3, '두 얼굴의 유사도': '{distance:.2f}', '일치 여부': True},
            ('임계값: 0.4, 두 얼굴의 유사도: 0.99', False) ]}]

    ```