from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

def convert(x):
    return max(0, int(x.numpy()))

def store(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size, target):
    model.eval()

    # Get dataloader
    # dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )
    classes = load_classes(opt.class_path)
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    img_label = 1
    for batch_i, (path, imgs) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        # sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        # outputs : detections
        if outputs is not None:
            img = np.array(Image.open(path[0]))
            for output in outputs:
                output = rescale_boxes(output, opt.img_size, img.shape[:2])
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in output:
                    if int(cls_pred.numpy())==target: # pereson=0
                        x1, y1, x2, y2 = convert(x1), convert(y1), convert(x2), convert(y2)
                        store_img = img[y1:y2,x1:x2]
                        plt.imsave(f'./store/{img_label}.jpg', store_img)
                        img_label += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default="config/yolov4.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov4.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--use_custom", type=bool, default=False, help="trained weight")
    parser.add_argument("--image_folder", type=str, default="data/sample_images", help="path to dataset")
    parser.add_argument("--target", type=int, default=0, help="target image to store")
    opt = parser.parse_args()

    # Use custom weight
    if opt.use_custom:
        opt.model_def = 'config/yolov4-custom.cfg'
        opt.class_path = 'data/custom/classes.names'
        opt.data_config = 'config/custom.data'
        ls = sorted(os.listdir('./weights/custom'))
        if len(ls) > 0:
            opt.weights_path = 'weights/custom/'+ls[-1]
        
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
        
    store(
        model,
        path=opt.image_folder,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        target=opt.target
    )
