from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
num=0


def max_light(light):
    #print('light',light)
    for i in range(0,len(light)):
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in light[i]:
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
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/sample_images", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov4.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov4.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_custom", type=bool, default=False, help="trained weight")
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
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
        og_img = Image.open(path)
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            #print(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            light=[]
            #    bbox
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                #print('detect',detections)
                #print(type(detections))
                if(int(cls_pred)==0 and cls_conf.item()>0.99 and cls_conf.item()<0.998): #light
                    light.append(torch.FloatTensor([[x1, y1, x2, y2, conf, cls_conf, cls_pred]]))
                    
                elif(int(cls_pred)==1):#crosswalk
                    color=(0.9058823529411765, 0.7294117647058823, 0.3215686274509804, 1.0)
                    box_w = x2 - x1
                    box_h = y2 - y1
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
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
                    )
                    
                    
            if len(light)>0: 
                x1, y1, x2, y2=max_light(light)
                box_w = x2 - x1
                box_h = y2 - y1
                color=(0.6470588235294118, 0.3176470588235294, 0.5803921568627451, 1.0)
                
                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
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
                )
                num_img = np.array(og_img)
                height, width = num_img.shape[:2]
                ROI_image = num_img[:120, :]
                light = False
                ROI_gray = cv2.cvtColor(ROI_image, cv2.COLOR_BGR2GRAY)

                x=x1
                y=y1

                ROI=ROI_gray[int(y):int(y) + int(box_h), int(x):int(x) + int(box_w)]
                ROI_X = ROI.shape[1]
                ROI_Y = ROI.shape[0]
                light_position = -1
                for i in range(0,ROI_Y):
                    for j in range(0,ROI_X):
                        if ROI[i, j] > 100:
                            light_position = j
                if light_position > ROI_Y / 2:
                    plt.text(
                        x2,
                        y2,
                        s='green',  # classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                else:
                    plt.text(
                        x2,
                        y2,
                        s='red',  # classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )



        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split('\\')[-1].split(".")[0]
        plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
