# 1. Install
- 파일을 git을통해 내려받거나 zip으로 받은 후 requirements 를 다운
```
git clone https://github.com/minji-o-j/system-for-visually-impaired.git
cd system-for-visually-impaired
pip install -r requirements.txt
```

---
# 2. Download Prerequisite (YOLOv4)
- Darknet 파일 다운
- 다운 후 해당 파일을 weights 폴더에 위치시킨다.
```
# yolov4.conv.137
https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
# weights
https://drive.google.com/file/d/1cewMfusmPjYWbrnuJRuKhPMwRe_b9PaT/view
```

---
# 3. Pretrained test
- `data/sample_images`에 테스트 해보고 싶은 파일들을 위치시키고 다음과 같은 코드를 입력하면 YOLOv4 기본 classes에 대한 결과를 확인 가능
- output 폴더에 결과 파일 저장됨  
```py
!python detect.py
```

---
# 4. Custom Dataset

