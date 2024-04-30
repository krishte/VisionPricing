# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import numpy as np

import wandb
from wandb.integration.ultralytics import add_wandb_callback

import torch


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        torch.cuda.set_device(0)

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    # brand model:
    # train6: LG + KitchenAid
    # train12: Tejas' 10

    #Object model:
    # train: Coco128

    # Step 1: Initialize a Weights & Biases run
    # wandb.init(project="oxford_group_project", job_type="training", name="logo_detector2")

    # Create a new YOLO model from scratch
    model = YOLO('yolov8s.pt')
    # add_wandb_callback(model, enable_model_checkpointing=True)

    # Load a pretrained YOLO model (recommended for training)
    # brand_model = YOLO('runs/detect/train6/weights/best.pt')
    #object_model = YOLO('runs/detect/train/weights/best.pt')
    # Train the model using the 'coco128.yaml' dataset for 3 epochs

    results = model.train(project="oxford_group_project", data='brand_dataset.yaml', epochs=100, name="logo_detector" )

    # # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    #results = object_model('https://ultralytics.com/images/bus.jpg')

    # Export the model to ONNX format
    model.export(format='onnx')

    # wandb.finish()

    # test_image = "datasets/fridge_type/train/bottom_freezer/bf_1.jpg"


    # # brand_results = brand_model(test_image)
    # object_results = object_model(test_image)

    # print ("results", object_results[0].boxes)
    # box =  np.array(object_results[0].boxes.xyxy)[list(object_results[0].boxes.cls).index(72.)]

    # object_frame = object_results[0].plot()
    # # final_frame = brand_results[0].plot(img=object_frame)
    # image = cv2.imread(test_image)
    # cv2.imwrite("test.png", image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])
    #cv2.imwrite("test.png", object_frame)

if __name__ ==  '__main__':
    main()