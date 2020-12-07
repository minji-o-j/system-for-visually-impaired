<a href="https://www.youtube.com/watch?v=qxElChg70Ck">
  <img src="https://user-images.githubusercontent.com/45448731/101240773-bb971580-3734-11eb-8306-26d3920e593f.png"/>
</a>

▲클릭하시면 소개 영상을 볼 수 있습니다. 
<br><br>
 

[코드 실행 방법](https://github.com/minji-o-j/system-for-visually-impaired/blob/master/How%20to%20Use(%ED%95%9C%EA%B5%AD%EC%96%B4).md)

---
![HitCount](http://hits.dwyl.com/minji-o-j/system-for-visually-impaired.svg)
[　](https://github.com/ML-DL-Study/system-for-visually-impaired/compare/master...minji-o-j:master)

---
## 개발기간
20/04~20/12 

---
## 개발자
이름|Github|비고
----|---|---
김성현|[@Seong-Hyun-0224](https://github.com/Seong-Hyun-0224)|
김한솔|[@hansol0118](https://github.com/hansol0118)| 
윤대호|[@201810788](https://github.com/201810788)|~2020/06
이혜인|[@hyeinlee725](https://github.com/hyeinlee725)|~2020/08
정민지|[@minji-o-j](https://github.com/minji-o-j)|

---
## 프로그램 설명
### "시각장애인을 위한 도보환경 알리미"
1. 앞에 있는 횡단보도와 신호등을 **알아서** 찾아주고
2. 신호등의 색깔까지 **알아서** 알려주고
3. 횡단보도를 건너야 할 타이밍까지 **알아서** 알려주는  
시각장애인의 안전한 도보 환경 구축을 위한 서비스  

---
## 목적 및 필요성 
<img src="https://user-images.githubusercontent.com/61938029/101280932-6c71e300-380f-11eb-9f6b-617ac933f7ee.png" height="240px"/> <img src="https://user-images.githubusercontent.com/61938029/101280935-70056a00-380f-11eb-86fb-46cfe8078322.png" height="240px"/> <img src="https://user-images.githubusercontent.com/61938029/101280940-75fb4b00-380f-11eb-8fb5-95f6df5f094f.png" height="240px"/>
> 고장난 음향신호기, 찾기 힘든 음향신호기의 위치

-	시각장애인들은 점자 블록을 통해 횡단보도를 찾고, 음향 신호기를 통해 신호등의 불빛 색깔을 파악함

-	하지만 2020년 서울 강북구 시각장애인 음향신호기 조사 결과 부적합 판정 **45.9%** 정도로 제대로 설치되어 있지 않은 곳이 많음.

- 또한 한국 장애인 재활 협회 조사 결과 시각장애인 대부분이 **횡단보도 및 신호등, 음성안내 시스템 버튼의 위치 파악에 어려움**을 겪고 있으며, **횡단보도 횡단에 도움이 되는 다른 장치가 필요하다**고 응답함  

---
## 기대효과
- 횡단보도와 신호등이 있음을 파악함과 동사에 신호등의 색상을 검출하여 건너야 할 타이밍을 알려줌으로,써 시각장애인이 횡단보도 이용시 겪는 문제를 해결할 것으로 예상함

---
## 데이터 구축
### 1. 학습용 데이터 수집
- 약 **4000**장
  - Kaggle
  - google 크롤링 데이터
  - 직접 촬영한 data


### 2. 학습용 데이터 전처리(image resize)
- 모은 데이터들을 학습용 데이터로 사용하기 위한 크기 동일화.
- python을 이용한 이미지 크기 512x512 resize.
<img src="https://user-images.githubusercontent.com/61938029/101287681-16179b00-3835-11eb-8d5d-fad6f0e8c413.png" width="500px"/>

### 3. Data Labeling
- 각각의 이미지에 대해 횡단보도, 신호등 영역을 라벨링 해 위치 지정.
  - 횡단보도: crosswalk, 신호등: light  
- 라벨링한 이미지 대해 대응되는 4000개의 xml파일을 생성하는 **labelImg** 프로그램 이용.
<img src="https://user-images.githubusercontent.com/61938029/101287684-1fa10300-3835-11eb-8dd5-9203e5e91e76.png" width="500px"/>

### 4. Dataset 구성
- 위 과정들을 통해 resize된 4000장의 이미지 파일(image)과 대응되는 4000장의 xml 파일(annotation)이 최종 Dataset.
- 80%: 학습용 데이터(train)로 사용 / 20%: 검증용 데이터(val)로 사용.
![image](https://user-images.githubusercontent.com/61938029/101287692-2fb8e280-3835-11eb-933b-05e6183fd339.png)

---
## 사용 프로그램
![image](https://user-images.githubusercontent.com/45448731/101232065-c5982480-36f2-11eb-894f-bb80c7f722a4.png)

---
## 사용 모델
### 1차 모델 (20/04~20/05): [YOLOv2](https://github.com/minji-o-j/system-for-visually-impaired/tree/master/v.1.0_YOLOv2(~200529))

<img src="https://user-images.githubusercontent.com/61938029/101282894-f07d9800-381a-11eb-8383-0566207232e1.png" width="300px"/>   <img src="https://user-images.githubusercontent.com/61938029/101282941-2e7abc00-381b-11eb-9a7d-39cd680fa0c2.png" width="300px"/>

- 위 사진처럼 성능과 정확도가 좋지 않아 다른 모델을 찾기로 함.

<br>

### 2차 모델(20/08~20/12) : YOLOv4

![image](https://user-images.githubusercontent.com/61938029/101286675-24fb4f00-382f-11eb-8430-da043070b18c.png)

- YOLOv4: 최신기술들의 조합을 통해 만들어져 최적의 속도와 정확도를 자랑하는 모델.
  - YOLOv3 / CSPDarknet53 / SPP / PAN / BOF / BOS 등의 최신 기술 사용.
  - YOLOv2, v3에서 작은 object에 대해 인식하지 못하는 것에 대해 Bag or Freebies(BOF) / Bag of Specials(BOS) 2가지 유형의 기술을 적용하여 해결.

#### BOF
- Data augmenation, Loss function, Regularization 등 학습에 관여하는 요소로, training cost를 증가시켜서 정확도를 높이는 방법들을 의미.
 
#### BOS
- architecture 관점에서의 기법들이 주를 이루고, post processing도 포함되어 있으며, 오로지 inference cost만 증가시켜서 정확도를 높이는 기법들을 의미.
  
#### YOLOv4 architecture
- **YOLOv4** = VOLOv3 기반 + backbone(CSPDarkNet53) + Neck(SPP, PAN) + BOF, BOS 기법 적용.

---
## 알고리즘
### 신호등 색상 검출 알고리즘
- HSV color 모델 사용  

<img src="https://user-images.githubusercontent.com/61938029/101289435-bc689e00-383f-11eb-8c48-c7e587f6c47c.png" width="400px" height="200px"/> <img src="https://user-images.githubusercontent.com/61938029/101289440-cbe7e700-383f-11eb-818a-69d8d80e0596.png" width="400px" height="200px"/>
> HSV이미지, H, S, V 이미지 

- 신호등이 검출되었을 때 **밝기(V) 이미지**가 신호등의 불빛을 뚜렷하게 나타냄

- 검출된 신호등의 영역 중 위(빨간불 영역)/아래(초록불 영역)를 1/2~1/4 지점으로 나누어 픽셀 값 더하여 어떤 영역에 밝기값 더 많은지 판단
  - 박스 size에 따라서 나누는 기준 다르게 설정함  

### 도보 환경 알리미 알고리즘  
![image](https://user-images.githubusercontent.com/61938029/101287890-62170f80-3836-11eb-96bf-f6cd5fd0fb55.png)  
 1. **state = -2**: 신호등 검출 진행 중
 2. **state = -1**: 신호등 검출 진행 중
 3. **state = 0**: 신호등 없음
 4. **state = 1**: 초기 초록불 검출됨. **기다리라**는 문구 출력.
 5. **state = 2**: 빨간불 검출 혹은 빨간불로 바뀜. **기다리라**는 문구 출력.
 6. **state = 3**: 건너도 되는 초록불. **건너도 된다**는 문구 출력.

---
## 시연 영상
#### 클릭하시면 유튜브로 연결됩니다!
### 초기 초록불 인식된 경우
<a href="https://www.youtube.com/watch?v=AHQ358YE6tY">
  <img src="https://user-images.githubusercontent.com/61938029/101289718-93e1a380-3841-11eb-81cc-546f7c786f71.png" width="700px"/>
</a>

### 초기 빨간불 인식된 경우
<a href="https://youtu.be/q3KLPQ416m8">
  <img src="https://user-images.githubusercontent.com/61938029/101289753-c7243280-3841-11eb-9f81-f062c24020eb.png" width="700px"/>
</a>


