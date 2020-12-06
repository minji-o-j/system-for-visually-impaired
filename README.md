<a href="https://www.youtube.com/watch?v=qxElChg70Ck">
  <img src="https://user-images.githubusercontent.com/45448731/101240773-bb971580-3734-11eb-8306-26d3920e593f.png"/>
</a>

▲클릭하시면 소개 영상을 볼 수 있습니다. 
<br><br>
 

[코드 실행 방법](https://github.com/minji-o-j/system-for-visually-impaired/blob/master/How%20to%20Use(%ED%95%9C%EA%B5%AD%EC%96%B4).md)

---
![HitCount](http://hits.dwyl.com/minji-o-j/system-for-visually-impaired.svg)
[　](https://github.com/ML-DL-Study/system-for-visually-impaired/compare/master...minji-o-j:master)


## 개발기간
20/04~20/12 

## 개발자
이름|Github|비고
----|---|---
김성현|[@Seong-Hyun-0224](https://github.com/Seong-Hyun-0224)|
김한솔|[@hansol0118](https://github.com/hansol0118)| 
윤대호|[@201810788](https://github.com/201810788)|~2020/06
이혜인|[@hyeinlee725](https://github.com/hyeinlee725)|~2020/08
정민지|[@minji-o-j](https://github.com/minji-o-j)|


## 목적 및 필요성 


-	시각장애인들은 점자 블록을 통해 횡단보도를 찾고, 음향 신호기를 통해 신호등의 불빛 색깔을 파악함
-	하지만 2020년 서울 강북구 시각장애인 음향신호기 조사 결과 부적합 판정 **45.9%** 정도로 제대로 설치되어 있지 않은 곳이 많음.

![image](https://user-images.githubusercontent.com/61938029/101280932-6c71e300-380f-11eb-9f6b-617ac933f7ee.png)
![image](https://user-images.githubusercontent.com/61938029/101280935-70056a00-380f-11eb-86fb-46cfe8078322.png)
![image](https://user-images.githubusercontent.com/61938029/101280940-75fb4b00-380f-11eb-8fb5-95f6df5f094f.png)


# 복붙해서 엔터치지말고 글을 읽고 정리해줘...다시 손 안가게끔제발..ㅠㅠ

## 기대효과

- 한국장애인재활협회의 시각 장애인을 대상으로 한 설문 조사 결과 
  - 횡단보도와 신호등 위치를 예측하기, 음향 신호기의 위치 찾기, 신호등의 녹색 적색 여부를 알기 어려움 등의 문제 존재.
  
- 도보 환경 알림이 서비스는 이러한 점들을 해결
  - 음향 신호기 버튼을 누르지 않아도 횡단보도, 신호등, 그리고 신호등의 색상을 검출
  - 앞서 시각장애인이 겪는 문제들을 크게 해소할 수 있을 것이라 예상.

- 또한 본 서비스에 보행 시 부딪힐 위험이 있는 장애물들을 검출하는 기능 등을 추가
  - 시각장애인 보행 문제 뿐 아니라 다른 문제해결에도 적용가능.
  
- 이외에도 본 기술을 자율 주행 서비스에 적용하는 등 더 넓은 범위에도 적용할 수 있음.

## 사용 프로그램
![image](https://user-images.githubusercontent.com/45448731/101232065-c5982480-36f2-11eb-894f-bb80c7f722a4.png)
---
## 사용 모델
### 1차 모델 (20/04~20/05): [YOLOv2](https://github.com/minji-o-j/system-for-visually-impaired/tree/master/v.1.0_YOLOv2(~200529))

<img src="https://user-images.githubusercontent.com/61938029/101282894-f07d9800-381a-11eb-8383-0566207232e1.png" width="400px"/>   <img src="https://user-images.githubusercontent.com/61938029/101282941-2e7abc00-381b-11eb-9a7d-39cd680fa0c2.png" width="400px"/>

- 설명

<br>

### 2차 모델(20/08~20/12) : YOLOv4
- YOLOv4모델에 대한 설명 적는다
ppt에 있는 yolov4에 결합되었던것들 언급후적기 : 우리 모델 설명을 영상에서 자세히 안해서 여기서 좀 자세히?

---
## 알고리즘
### 신호등 색상 검출 알고리즘  
### 도보 환경 알리미 알고리즘  
---
## 시연 영상
- 유튜브 링크 이미지로바꿔서 넣기  
https://www.youtube.com/watch?v=AHQ358YE6tY  
https://youtu.be/q3KLPQ416m8  
