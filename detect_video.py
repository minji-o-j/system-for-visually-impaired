from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2
import numpy as np

class CircularQueue:

    #큐 초기화
        def __init__(self, n):
            self.maxCount = n
            self.data = [0] * n
            self.count = 0
            self.front = -1
            self.rear = -1

        # 현재 큐 길이를 반환
        def size(self):
            return self.count

        # 큐가 꽉 차있는지
        def isFull(self):
            return self.count == self.maxCount
        
        # 큐가 비어있는지
        def isEmpty(self):
            return self.count==0
        
        # 데이터 원소 추가
        def enqueue(self, x):
            if self.isFull():
                #raise IndexError('Queue full')
                self.dequeue()

            self.rear = (self.rear + 1) % self.maxCount
            self.data[self.rear] = x
            self.count += 1

        #데이터 원소 제거
        def dequeue(self):
            if self.isEmpty():
                raise IndexError('Queue empty')

            self.front = (self.front + 1) % self.maxCount
            x = self.data[self.front]
            self.count -= 1
            return x

        # 큐의 맨 앞 원소 반환
        def peek(self):
            if self.isEmpty():
                raise IndexError('Queue empty')
            return self.data[(self.front + 1) % self.maxCount]
        
        def makeEmpty(self):
            for i in range(0,self.maxCount):
                self.dequeue()
                
        def display(self):
            print(self.data)
        
        def return_data(self):
            return self.data
            
def max_light(light):
    #print('light',light)
    for i in range(0,len(light)):
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in light[i]:
            print(x1, y1, x2, y2)
            if x1<0:
                x1=0
            if y1<0:
                y1=0
            if i==0: #초기선언
                min_x1 = x1
                max_x2 = x2
                min_y1 = y1
                max_y2 = y2
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            if (min_x1 > x1):
                min_x1 = x1
            if (max_x2 < x2):
                max_x2 = x2
            if (min_y1 > y1):
                min_y1 = y1
            if (max_y2 < y2):
                max_y2 = y2
    return min_x1,min_y1,max_x2,max_y2

def light_color(x1, y1, x2, y2, path):

    image=cv2.imread(path) #BGR
    image_ycbcr=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

    cut_light=image_ycbcr[int(y1):int(y2),int(x1):int(x2),:]

    if x2-x1>130:
        cut_d2=int(cut_light.shape[0]/4)
    else:
        cut_d2=int(cut_light.shape[0]/4.3)
    '''
    color='orange'
    plt.text(
    x1,
    y1+cut_d2,
    s= 'red line',
    color="white",
 verticalalignment="top",
    bbox={"color": color, "pad": 0},
    )
    '''
    cut_top=cut_light[0:cut_d2,:,:]
    cut_bottom=cut_light[cut_d2:cut_light.shape[0],:,:]
    
    H1=0
    S1=0
    V1=0
    H2=0
    S2=0
    V2=0
    
    #top
    for i in range(0,cut_top.shape[0]): #x
        for j in range(0,cut_top.shape[1]): #y
            #print(cut_top[i][j][0],cut_top[i][j][1],cut_top[i][j][2])
            H1+=cut_top[i][j][0]
            S1+=cut_top[i][j][1]
            V1+=cut_top[i][j][2]
            
            
    #bottom
    for i in range(0,cut_bottom.shape[0]): #x
        for j in range(0,cut_bottom.shape[1]): #y
            H2+=cut_bottom[i][j][0]
            S2+=cut_bottom[i][j][1]
            V2+=cut_bottom[i][j][2]
            
    V1size=V1/(cut_top.shape[0]*cut_top.shape[1])
    V2size=V2/(cut_bottom.shape[0]*cut_bottom.shape[1])
    
    print(V1size,V2size)
    if(V1size>V2size):
        l_color='red'
        selected=V1size
    else:
        l_color='green'
        selected=V2size
    return l_color, selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/sample_videos/sample", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov4.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov4.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_custom", type=bool, default=False, help="trained weight")
    parser.add_argument("--export_video_frame",default=False,type=bool) #비디오 프레임부터 꺼내야할때
    opt = parser.parse_args()

    # Use custom weight
    if opt.use_custom:
        opt.model_def = 'config/yolov4-custom.cfg'
        opt.class_path = 'data/custom/classes.names'
        ls = sorted(os.listdir('./weights/custom'))
        if len(ls) > 0:
            opt.weights_path = 'weights/custom/'+ls[-1]
        
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    ##-----video
    if(opt.export_video_frame==True):
        vidcap = cv2.VideoCapture('data/sample_videos/rg.mp4')
        def getFrame(sec,imgarr):
            vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
            hasFrames,image = vidcap.read()
            if hasFrames:
                cv2.imwrite(os.path.join(opt.image_folder, str(count) + '.png'), image)     # save frame as png file
            return hasFrames

        sec = 0
        frameRate = 0.042 #//it will capture image in each 0.04 second
        count=1
        imgarr=[]
        success = getFrame(sec,imgarr)

        num=0
        while success:
            if num%100==0:
                print(num)
            num+=1
            count = count + 1
            sec = sec + frameRate
            sec = round(sec, 2)
            success = getFrame(sec,imgarr)

        print("frame: "+str(num))
    ##-----

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    #print(dataloader)
    

    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):

        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
   

    # ------------------------------------------------------------------------------------------
    state=-2
    
    # 횡단보도 queue
    crosswalk_queue=CircularQueue(24)
    
    # 신호등 queue=[]
    light_queue=CircularQueue(24)

    
    #이미지 정렬---------
    ##0배열 선언
    imgs_n=[0 for i in range(len(imgs))]
    img_detections_n= [0 for i in range(len(img_detections))]
    #print(len(imgs),len(img_detections))
    for i in range(0,len(imgs)):
        try:
            try:#4자리
                cutstr=imgs[i][-8:]
                cutstr=cutstr[0:4] #앞에4개숫자 자름
                cutint=int(cutstr)
                imgs_n[cutint-1]=imgs[i]
                img_detections_n[cutint-1]=img_detections[i]
            except:#3자리
                cutstr=imgs[i][-7:]
                cutstr=cutstr[0:3] #앞에3개숫자 자름
                cutint=int(cutstr)
                imgs_n[cutint-1]=imgs[i]
                img_detections_n[cutint-1]=img_detections[i]
        except:
            try: #2자리
                cutstr=imgs[i][-6:]
                cutstr=cutstr[0:2] #앞에2개숫자 자름
                cutint=int(cutstr)
                imgs_n[cutint-1]=imgs[i]
                img_detections_n[cutint-1]=img_detections[i]
            except:#1자리
                cutstr=imgs[i][-5:]
                cutstr=cutstr[0:1] #앞에2개숫자 자름
                cutint=int(cutstr)
                imgs_n[cutint-1]=imgs[i]
                img_detections_n[cutint-1]=img_detections[i]
            
    for img_i, (path, detections) in enumerate(zip(imgs_n, img_detections_n)): #imgs: 위치, img_detections:목록, img_i 인덱스
        print('testqueue:',crosswalk_queue.isFull(),light_queue.isFull())
        print("(%d) Image: '%s'" % (img_i, path))
        #print("Image: '%s'" % (path))
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        #plt.figure(figsize=(10,5.625)) #plot size
        fig, ax = plt.subplots(figsize=(32,18))#) ############################################################size
        #print("fig,ax-----")
        #print(fig,ax)
        #plt.figure(figsize=(10,5.625)) #plot size
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            #print(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            light=[]
            crosswalk=[]
            #    bbox
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #print('detect',detections)
                #print(type(detections))
                if(int(cls_pred)==0 and cls_conf.item()>0.98  and cls_conf.item()<1): #light9957 9975 $9941
                    light.append(torch.FloatTensor([[x1, y1, x2, y2, conf, cls_conf, cls_pred]]))
                    
                elif(int(cls_pred)==1): #crosswalk
                    if cls_conf.item()>0.8:
                        color=(0.9058823529411765, 0.7294117647058823, 0.3215686274509804, 1.0)
                        # 이미지 벗어남 방지
                        if x1<0:
                            x1=0
                        if y1<0:
                            y1=0
                        box_w = x2 - x1
                        box_h = y2 - y1
                        print(x1,y1,x2,y2)
                        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                        # Create a Rectangle patch
                        bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=4, edgecolor=color, facecolor="none")
                        # Add the bbox to the plot
                        ax.add_patch(bbox)
                        # Add label
                        plt.text(
                            x1,
                            y1,
                            s=classes[int(cls_pred)],
                            color="white",
                            verticalalignment="top",
                            bbox={"color": color, "pad": 0},
                            fontsize=30,
                        )
                        crosswalk.append(1) #crosswalk 검출되었음(여러개 검출될 수도 있어서 넣은 것)
            
            # 횡단보도 검출 전일 때 큐에 추가
            if state==-2 and len(crosswalk)>0:
                crosswalk_queue.enqueue(1) #꽉차면 알아서 맨앞 값이 빠지고 맨뒤에 들어간다
                
                
            #횡단보도 찾고있는 중이나 검출이 안됨
            elif state==-2 and len(crosswalk)==0:
                crosswalk_queue.enqueue(0)
                
            #횡단보도 검출 된 상태인지 검사, 24frame이 다 들어왔을때만 검사한다
            if state==-2 and crosswalk_queue.isFull():
                findcrosswalk=crosswalk_queue.return_data()
                if findcrosswalk.count(1)>15:
                    state=-1
                    print('find crosswalk')
                
                
            
            
            #light 검출이 되어 박스 그릴거임
            if len(light)>0: 
                x1, y1, x2, y2=max_light(light)
                box_w = x2 - x1
                box_h = y2 - y1
                color=(0.6470588235294118, 0.3176470588235294, 0.5803921568627451, 1.0)
                


                if box_h>187 or box_h<90 or box_w >120: #수치조정필요 #신호등이 검출되었으나 신호등이 큰 경우
                    #147 or box_h<90 or box_w >85
                    print('big',box_h,box_w)
                    '''
                    # 수치 조정 위해서 임시로 넣어놓음
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    plt.text(
                        x2,
                        y2,
                        s=light_color(x1, y1, x2, y2,path),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                    '''
                    
                    
                else: #적당한 size신호등 검출
                # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=4, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    
                # Add label
                    plt.text(
                        x1,
                        y1,
                        s='light',#classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                        fontsize=30,
                    )
                    
                    l_color,color_value =light_color(x1, y1, x2, y2,path) #신호등 색상 검출
                    if (state==-1 or state==1 or state==2) and l_color=='green':
                        light_queue.enqueue(1)
                        
                    elif (state==-1 or state==1 or state==2) and l_color=='red':
                        light_queue.enqueue(2)
                    
                    #########색상안보이게
                    '''
                    #print('test',x1,light_position_x,y1,light_position_y)
                    plt.text(
                        x2,
                        y2,
                        s=light_color(x1, y1, x2, y2,path),
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                    '''

            
            #light 검출 안됨
            if state==-1 and len(light)==0:
                light_queue.enqueue(0) #꽉차면 알아서 맨앞 값이 빠지고 맨뒤에 들어간다
                
             
                        
            # 신호등 state 검사
            if state==-1 and light_queue.isFull(): #신호등 찾는중
                findlight=light_queue.return_data()
                        
                if findlight.count(1)+findlight.count(2)<6: #0이 16개이상, 신호등 없음 #일단설정
                    state=0
                    print('no light')
                    
                else:
                    if findlight.count(1)>findlight.count(2): #green
                        state=1
                        #light_queue.makeEmpty()
                        
                    else: #red
                        state=2
                        #light_queue.makeEmpty()
                        
            elif state==1: #초기초록불
                if findlight.count(1)<findlight.count(2): #초<빨
                    #print('findred...',findlight.count(2),findlight.count(1)) #r,g
                    state=2
                    print('turn red')
                    
            elif state==2: #빨간불            
                if findlight.count(1)>findlight.count(2): #초>빨
                    state=3
                    print('turn green')
                  
            
            #state 표시
            if state==-2: #횡단보도 검출 전
                color=(0.0, 0.0, 0.0, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'finding crosswalk...',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
                
            elif state==-1: # 신호등 검출 진행중
                color=(0.0, 0.0, 0.0, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'finding signal...',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
               
            elif state==0: #신호등 없음
                color=(0.0, 0.0, 0.0, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'no signal',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
                
            elif state==1: # 초기 신호 초록불
                color=(0.0, 1.0, 0.1, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'Please wait next green signal',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
                
            elif state==2: # 신호 빨간불
                color=(1.0, 0.0, 0.0, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'please wait green signal',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
                
            elif state==3: # 바뀐 신호 초록불(건너도 됨)
                color=(0.0, 1.0, 0.0, 1.0)
                plt.text(
                    0,
                    0,
                    s= 'Cross the road!',
                    color="white",
                 verticalalignment="top",
                    bbox={"color": color, "pad": 0},
                    fontsize=30,
                )
                
            #shape 확인용
            #break;
        # Save generated image with detections
        
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split('\\')[-1].split(".")[0]
        plt.savefig(f"output/test1/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        
    imgarr=[]
    path_dir = 'output/test1/'
    file_list = os.listdir(path_dir)
    
    #숫자 이름대로 정렬
    #str to int
    for i in range(len(file_list)):
        file_list[i]=int(file_list[i].replace(".png","")) #testlist에서 ".png" 제거, 정수로 변환
    file_list.sort()
    
    for i in range(len(file_list)):
        file_list[i]=str(file_list[i])+".png"
    
    #print(file_list) #숫자 순서대로 정렬된 것 확인함
    
    for png in file_list:
        #image = Image.open(path_dir + png).convert("RGB")
        image=cv2.imread(path_dir + png)
        pixel = np.array(image)
        #print(np.shape(pixel))
        #pixel2=np.delete(pixel, 3, axis = 2)
        
        if(np.shape(pixel)!=(1386,2464, 3)): #1386,2464
            #print("hello")
            pixel=pixel[0:1386,0:2464,0:3]
            '''
        if(np.shape(pixel)!=(437, 779, 3)):
            print(np.shape(pixel))
            pixel=pixel[0:437,0:779,0:3]
        '''
        #print(np.shape(pixel2))
        imgarr.append(pixel)
    #print(np.shape(imgarr))

        

    fps = 24 #24 #frame per second
    pathOut = 'output/rg.mp4'
    size=(2464,1386)#(2464,1386)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(imgarr)):
        # writing to a image array
        out.write(imgarr[i])
        #print(imgarr[i])
    
    out.release()