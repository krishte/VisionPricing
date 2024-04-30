The following is an explanation of the various components of this repo.

## Datasets
This contains all datasets used for training classification and detection models. The **brand_dataset** contains all brand logo images and labels and **brand_dataset_augmented** is the former with random data augmentations applied. The **extra_objects** dataset is images for 6 classes of electronics products not included in the standard COCO dataset YOLO is trained on. For each object specific model that needs to be trained, there is an **object_specific_model** and a **object_specific_model_real**. The former contains all images downloaded from google and the latter contains the images cropped to just the relevenat object.

## Runs + oxford_group_project
All past model trainings are stored here including various statistical measures and model weights. **runs** is for cpu trainings and **oxford_group_project** for gpu trainings.

## Test_images
Sample images for testing the final model (stored here just for convenience)

## classifier.py
The final program which combines the object detection model, logo detection model, object specific classication models, and a k-means based dominant color finding program to output a description of the product

## yolo.py
A very basic script for model training. Detection models require a yaml file like **brand_dataset.yaml** to specify details about data locations and labels. **brand_dataset.yaml** is for training using **brand_dataset_augmented** and **object_dataset.yaml** is for training using **extra_objects**l. Classification models just require the images to be split up into folders based on the class.

## labelEditer.py
Contains a few small scripts to handle some processing tasks numbered as follows:
1. Shifts all the class values in the label .txt files of a YOLO dataset
2. Crops all images to just the object specified using the object detection model
3. Outputs a list of class label numbers to class label strings from the yaml file
4. Converts webp to png images

## renamer.py
A script for renaming all images into a standard format

## openimages_downloader.py
Downloads the images and labels for the 6 extra object classes and converts the dataset to YOLO format into **datasets/extra_objects**

## Pipenv
I use pipenv for python package management, so downloading pipenv will make installing all required dependencies very easy. Everything is currently running on python 3.10.5
