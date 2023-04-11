- 기본 트리 구조입니다.

```
    AI_16_CP2
    ├─controller
    ├─function
    ├─models
    └─service
```

- 사용법
    ```
        from <AI_16_CP2까지의 경로>.face_ds_project import FaceDSProject

        # project = FaceDSProject(min_detection_confidence = 0.2, model = 'vggface', distance_metric = 'cosine') # 이와 같이 패러미터를 구성할 수 있습니다. 이상은 디폴트값으로, 아래와 같습니다.
        project = FaceDSProject()

        image_path1 = 'path/to/image' # 시스템 경로
        image_path2 = 'http://<path/to/image>' # url주소
        image_path3 = [[[255, 255, 255]
                        [255, 255, 255]
                        [255, 255, 255]
                        # ...
                        [255, 255, 255]
                        [255, 255, 255]
                        [255, 255, 255]]] # numpy.ndarray(RGB값)
        
        # 
        face_img_list = project.get_faces('<image_path>') 