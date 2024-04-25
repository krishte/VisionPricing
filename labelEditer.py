from os import listdir
from os.path import isfile, join
import yaml
from PIL import Image
import os
from ultralytics import YOLO
import cv2
import numpy as np



### Shift all the label values in a YOLO dataset

# path = "cringe_better"
# onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

# for filename in onlyfiles:
#     data = None
#     with open('{}'.format(path + "/" + filename), 'r') as file:
#         data = file.read()
#         lines = data.split('\n')
#         for i, line in enumerate(lines):
#             ints = line.split(' ')
#             ints[0] = str(int(ints[0])+10)
#             lines[i] = ' '.join(ints)
#         new_data = '\n'.join(lines)
#         with open('{}'.format(path + "/" + filename), 'w') as file:
#             file.write(new_data)

### Data preprocessing: cropping object images to object

# types = ["4-slice", "2-slice"]
# not_found = []
# object_index = 70.
# item = "toaster"
# dataset = "toaster_type"

# for fridge_type in types:
#     path = "datasets/" + dataset + "/" + fridge_type
#     onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]

#     object_model = YOLO('runs/detect/train/weights/best.pt')


#     for i,file in enumerate(onlyfiles):
#         image_path = path + "/" + file
#         object_results = object_model(image_path)

#         try:
#             print(object_results[0].boxes.cls)
#             box =  np.array(object_results[0].boxes.xyxy)[list(object_results[0].boxes.cls).index(object_index)]
#         except:
#             print("no " + item + " found in image " + file)
#             not_found.append(fridge_type + " " + file)
#             continue

#         image = cv2.imread(image_path)
#         cv2.imwrite("datasets/" + dataset + "_real/train/"+ fridge_type + "/" + fridge_type + "_" + str(i) +".png", image[int(box[1]):int(box[3]), int(box[0]):int(box[2])])

# print(not_found)

### Print a dictionary of label numbers to label strings from yaml
            
# with open("brand_dataset.yaml", 'r') as stream:
#     data_loaded = yaml.safe_load(stream)
# print(data_loaded)

### Convert webp to png 

path = "images"

# only_folders = [f for f in listdir(path)]

# for folder in only_folders:
only_files = [f for f in listdir(path)]
for filename in only_files:
    if (filename.split('.')[-1] == 'webp'):
        im = Image.open(path + "/" + filename).convert('RGB')
        im.save(path + '/' + filename.split('.')[0] + ".png", "png")
        os.remove(path + "/" + filename)
