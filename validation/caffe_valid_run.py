#!/usr/bin/python
import cv2
import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("input_txt", help="specify the validation txt path")
parser.add_argument("model", help="specify the path to prototxt file that will be used by our Caffe main.py")
parser.add_argument("weights",help="specify the path to caffemodel file that will be used by our Caffe main.py")

args = parser.parse_args()

input_txt = args.input_txt # 2012_val.txt
model = args.model
weights = args.weights

# each validation data
f = open(input_txt,"r") 
for image_path in f: 
    image_path = image_path.strip()
    image_jpg = image_path.split("/")[-1]
    image_id = image_jpg.split(".")[0]
    image_ext = image_jpg.split(".")[1]
    print ("Image to infer:%s"%image_id)
    print ("Reading images in: {0}").format(image_path)
    print ("Model to use: {}").format(model)
    print ("Weights to use: {}").format(weights)
    print ("Running Caffe inference...")
    subprocess.call(["python","yolo_main.py","-m",model,"-w",weights,"-i",image_path])
