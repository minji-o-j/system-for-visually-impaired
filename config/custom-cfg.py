#!/usr/bin/env python3
import os
import argparse



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_classes", '-n', type=int, default=1, help="number of classes")
    opt = parser.parse_args()

    n_class = opt.n_classes
    custom_num = 3*(n_class+5)
    f = open(f'./yolov4-custom.txt', 'r')
    tar = open(f'./yolov4-custom.cfg', 'w')

    convert_dict = {
        'custom_num': str(custom_num),
        '$NUM_CLASSES': str(n_class),
        'custom_max_batches': 2000*n_class,
        'min_steps': int(2000*n_class * 0.8),
        'max_steps': int(2000*n_class * 0.9),
    }
    a=f.read()
    for key in convert_dict.keys():
        a = a.replace(key, str(convert_dict[key]))

    tar.write(a)