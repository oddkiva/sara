#!/bin/bash
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4.weights
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
mv yolov4-tiny.weights trained_models/yolov4-tiny
mv yolov4.weights trained_models/yolov4
