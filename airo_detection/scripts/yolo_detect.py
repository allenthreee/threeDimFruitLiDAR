from ultralytics import YOLO
import sys, os
import cv2
import numpy as np
from geometry_msgs.msg import Point
import torch
from ultralytics import YOLO
yolo_test_path = os.path.dirname(__file__)
sys.path.append(yolo_test_path)
# Create a new YOLO model from scratch
# model = YOLO("yolov8n.yaml")
# print(f"before we load the model and the image\n\n")
# Load a pretrained YOLO model (recommended for training)
# Load a pretrained YOLO model (recommended for training)/home/allen/icuas24_ws_mini/src/airo_detection/scripts/last_yolo.pt

# model = YOLO("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/last.pt")

from darknet_ros_msgs.msg import BoundingBoxes, BoundingBox
# Load the image
# image = cv2.imread("/home/allen/icuas24_ws_mini/src/airo_detection/scripts/detect_fruit_test/demo_data/red_yellow.png")

# Perform object detection on the image using the model

def yolo_detect(image, model):
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("yolo_detect() called")
    # Process results list
    yolo_fruit_yellows = []
    yolo_fruit_reds = []
    results = model.track(image, verbose=False)#.to(device=device)
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        class_indices = boxes.cls  # Class indices of each detected object
        confidences = boxes.conf  # Confidence scores of each detected object

        xywhs = boxes.xywh
        
        for xywh, class_index, confidence  in zip(xywhs, class_indices, confidences):
            class_name = result.names[int(class_index)]
            if(class_name == "yellow" and confidence > 0.5):
                # print("yolo detect one yellow")
                point = Point()
                point.x = float(xywh[0])
                point.y = float(xywh[1])
                point.z = float((xywh[2]+xywh[3])/2)
                yolo_fruit_yellows.append(point)
            if(class_name == "red"  and confidence > 0.5):
                # print("yolo detect one red")
                point = Point()
                point.x = float(xywh[0])
                point.y = float(xywh[1])
                point.z = float((xywh[2]+xywh[3])/2)
                yolo_fruit_reds.append(point)
                
    return yolo_fruit_yellows, yolo_fruit_reds
    # print(f"=======================================================")
    # result.show()  # display to screen
    # result.save(filename="result.jpg")  # save to disk