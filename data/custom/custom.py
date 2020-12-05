import os
import argparse

ls = os.listdir('./train/images')

train = open('./train/train.txt', 'w')
train_path = 'data/custom/train/images/'

for l in ls:
    item=train_path+l
    train.writelines(item+'\n')

ls = os.listdir('./valid/images')
valid = open('./valid/valid.txt', 'w')
valid_path = 'data/custom/valid/images/'

for l in ls:
    item=valid_path+l
    valid.writelines(item+'\n')