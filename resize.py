import glob # file 접근을 용이하게 해주는 모듈
import cv2  # 각종 영상처리 api를 제공해줌

files = glob.glob("lights/*") #lights 폴더에 있는 모든 파일의 경로를 읽어와 리스트에 저장한다.
imgName = 'light'  # resize한 이미지의 이름을 설정
filenumber = 0     # resize한 이미지의 이름에 번호를 부여

for f in files:    # f가 순서대로 한번씩 files 리스트안에 있는 모든 요소로 선언된다. => f가 lights에 저장된 모든 파일의 경로가 되어준다. (0번 인덱스의 파일부터 마지막까지 순서대로)
    img = cv2.imread(f)    # img 변수에 파일 경로 f에 있는 파일을 읽어서 저장한다.
    resized = cv2.resize(img, (512, 512), cv2.INTER_AREA)   # img에 있는 파일을 512,512 사이즈로 resize 해준다. 보간법으로 cv2.INTER_AREA를 사용. resize 한 결과를 resized에 저장
    cv2.imwrite('lights_resized//%s_%s.jpg'%(imgName,filenumber), resized)  # resize된 이미지를 lights_resized 폴더에 'imgName_filenumber'란 이름의 파일에 write한다. 그런 이름의 파일이 없으면 생성함.
    filenumber= filenumber+1  # filenumber값을 올려 resized된 파일이 서로 다른 이름을 가지게 한다.

print("resize finish")  # 모든 과정이 완료되었음을 표시
