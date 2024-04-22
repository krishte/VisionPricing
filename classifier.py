from os import listdir
from os.path import isfile, join
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from statistics import mode
import matplotlib.pyplot as plt

# brand model:
# detect/train6: LG + KitchenAid
# detect/train12: Tejas' 10

# Object model:
# detect/train: Coco128

# Fridge model:
# classify/train3 : 4 fridge typpes

# Mouse-wired model:
# classify/mouse_wired: wired or wireless

# Keyboard model
# classify/keyboard_size: tenkeyless, compact, or full size


### Pipeline for using models

image_path = "test_images/fridge_1.png"

object_indices = {72.: "refrigerator", 62.: "tv", 63.: "laptop", 64.: "mouse", 65.: "remote", 66.: "keyboard", 67.: "cell phone", 68.: "microwave", 69.: "oven", 70: "toaster", 78: "hair drier"}
brand_indices = {0: 'KitchenAid', 1: 'LG', 2: 'Samsung', 3: 'Whirlpool', 4: 'Toshiba', 5: 'Miele', 6: 'Logitech', 7: 'Panasonic', 8: 'Electrolux', 9: 'Sony', 10: 'Morphy Richards', 11: 'Sharp', 12: 'Philips', 13: 'Oppo', 14: 'MSI', 15: 'IFB', 16: 'Huawei', 17: 'Havells', 18: 'Haier', 19: 'Acer'}
object_specific_models = {"refrigerator": "runs/classify/train3/weights/best.pt", "mouse":"runs/classify/mouse_wired/weights/best.pt" }
product_description = {"product" : "", "brand": "", "type": "", "color": ""}

rgb_to_text = [("black", (0,0,0)), ("white", (255, 255,255)),
               ("red", (255,0,0)), ("lime", (0,255, 0)),
               ("blue", (0,0,255)), ("maroon", (128,0, 0)),
               ("yellow", (255,255,0)), ("olive", (128,128, 0)),
               ("cyan", (0,255,255)), ("green", (0,128, 0)),
               ("magenta", (255,0,255)), ("purple", (128,0, 128)),
               ("silver", (192,192,192)), ("teal", (0,128, 128)),
               ("gray", (128,128,128)), ("navy", (0,0, 128)),
               ("brown", (165, 42, 42)), ("pink", (255, 192, 203))]

object_model = YOLO('runs/detect/train/weights/best.pt')
brand_model = YOLO('runs/detect/train12/weights/best.pt')

object_results = object_model(image_path)
#object_frame = object_results[0].plot()

classes_found = list(object_results[0].boxes.cls)

for index, item in object_indices.items():
    if index in classes_found:
        #Cropping the image to just the product and setting product
        product_description["product"] = item
        box = np.array(object_results[0].boxes.xyxy)[classes_found.index(index)]
        prob = np.array(object_results[0].boxes.conf)[classes_found.index(index)]
        image = cv2.imread(image_path)
        # Blue border for object detected
        object_frame = cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), thickness=2) 
        # Blue background for object label
        object_frame = cv2.rectangle(object_frame, (int(box[0]), int(box[3])), (int(box[0]+300), int(box[3])-30) , (255, 0, 0), -1)
        # Object label
        object_frame = cv2.putText(object_frame, item + ": " + format(prob, ".2f")  , (int(box[0]), int(box[3])-5), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 255, 255), 2) 
        cropped_image =  image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        # checking if there is an object specific model and setting type using cropped image
        if item in object_specific_models:
            object_specific_model = YOLO(object_specific_models[item])
            object_specific_results = object_specific_model(cropped_image)
            # Black background for classes
            object_frame = cv2.rectangle(object_frame, (0,0), (250, 30+25*len(np.array(object_specific_results[0].probs.data))) , (0, 0, 0), -1)
            object_frame = object_specific_results[0].plot(img=object_frame)
            product_description['type'] = " ".join(object_specific_results[0].names[np.argmax(np.array(object_specific_results[0].probs.data))].split('_'))
        
        
        # Finding the brand of the product
        brand_results = brand_model(image)
        object_frame = brand_results[0].plot(img=object_frame)
        for brand_index in list(brand_results[0].boxes.cls):
            product_description['brand'] = brand_indices[int(brand_index)]
            break



        # Finding the color of the product
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        reshape = cropped_image_rgb.reshape((cropped_image_rgb.shape[0] * cropped_image_rgb.shape[1], 3))

        cluster = KMeans(n_clusters=5).fit(reshape)

        color = np.array(cluster.cluster_centers_[mode(cluster.labels_)])
        print(color)
        product_description['color'] = min([(np.linalg.norm(np.array(col)-color), name) for (name, col) in rgb_to_text])[1]



        cv2.imwrite("test.png", object_frame)

        print(product_description)
        break

