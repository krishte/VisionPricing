# -*- coding: utf-8 -*-

from ultralytics import YOLO
import cv2
import numpy as np

# from wandb.integration.ultralytics import add_wandb_callback
# import wandb

import torch


def main():
    device = "0" if torch.cuda.is_available() else "cpu"
    if device == "0":
        torch.cuda.set_device(0)

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
    
    model = YOLO('yolov8s.pt')
    # add_wandb_callback(model, enable_model_checkpointing=True)

    # results = model.train(project="oxford_group_project", data='brand_dataset.yaml', epochs=100, name="logo_detector" )
    model.tune(data='brand_dataset.yaml', epochs=30, iterations=300, optimizer='AdamW', plots=False, save=False, val=False)

    # wandb.finish()
    

if __name__ ==  '__main__':
    main()