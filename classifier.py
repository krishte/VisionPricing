from os import listdir
from os.path import isfile, join
from ultralytics import YOLO
import cv2
import numpy as np
from sklearn.cluster import KMeans
from statistics import mode
import matplotlib.pyplot as plt
import pytesseract
import nltk
from transformers import BlipProcessor, BlipForConditionalGeneration
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

image_path = "test_images/mice.jpg"

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def image_captioner(image) -> str:
    """Provides information about the image"""
    raw_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    inputs = blip_processor(raw_image, return_tensors="pt")
    out = blip_model.generate(**inputs)
    return blip_processor.decode(out[0], skip_special_tokens=True)



object_indices = {72.: "refrigerator", 62.: "tv", 63.: "laptop", 64.: "mouse", 65.: "remote", 66.: "keyboard", 67.: "cell phone", 68.: "microwave", 69.: "oven", 70: "toaster", 78: "hair drier"}
brand_indices = {0: 'KitchenAid', 1: 'LG', 2: 'Samsung', 3: 'Whirlpool', 4: 'Toshiba', 5: 'Miele', 6: 'Logitech', 7: 'Panasonic', 
                 8: 'Electrolux', 9: 'Sony', 10: 'Morphy Richards', 11: 'Sharp', 12: 'Philips', 13: 'Oppo', 14: 'MSI', 15: 'IFB', 
                 16: 'Huawei', 17: 'Havells', 18: 'Haier', 19: 'Acer', 20: "Apple", 21: "Asus", 22: "Bosch", 23: "Corsair", 24: "Dell", 
                 25: "HP", 26: "Nespresso", 27: "Razer", 28: "Russell Hobbs", 29: "Vivo", 30: "Babyliss",  31: "Siemens",  32: "Xiaomi",  
                 33: "Breville",  34: "Liebherr",  35: "Nokia",  36: "Lenovo",  37: "Kenwood",  38: "Motorola",  39: "Hamilton Beach", 40: "JBL"}

object_specific_models = {"refrigerator": "runs/classify/train3/weights/best.pt", "mouse":"runs/classify/mouse_wired/weights/best.pt" }
products = []

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
# brand_model = YOLO('runs/detect/logo_detector4/weights/best.pt')
brand_model = YOLO('oxford_group_project/logo_detector/weights/best.pt')

object_results = object_model(image_path)
#object_frame = object_results[0].plot()



classes_found = list(object_results[0].boxes.cls.cpu())
image = cv2.imread(image_path)
object_frame = cv2.imread(image_path)

print(image_captioner(image))
#Whether the intersection area of two boxes is > 0.5 of the area of either box
def intersecting_box(box1, box2):
    intersect_box = [max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])]
    if intersect_box[3]-intersect_box[1] < 0 or intersect_box[2]-intersect_box[0] < 0:
        return False
    area_intersect_box = (intersect_box[3]-intersect_box[1])*(intersect_box[2]-intersect_box[0])
    area_box1 = (box1[3]-box1[1])*(box1[2]-box1[0])
    area_box2 = (box2[3] - box2[1])*(box2[2]-box2[0])
    return area_intersect_box > 0.5*area_box1 and area_intersect_box > 0.5*area_box2

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)# get grayscale image


for index, item in object_indices.items():
    #select largest area box for overlapping boxes and corresponding confidence score
    if index in classes_found:
        #boxes and confidences for all objects with the right index
        classes_found_indices = [i for (i, val) in enumerate(np.array(classes_found)) if val==index]
        boxes = np.array(object_results[0].boxes.xyxy.cpu())[classes_found_indices]
        probs = np.array(object_results[0].boxes.conf.cpu())[classes_found_indices]
        nonintersecting_boxes, nonintersecting_probs = [], []

        for j,val in enumerate(boxes):
            found = False
            for i,box in enumerate(nonintersecting_boxes):
                if intersecting_box(box, val):
                    found = True
                    if probs[j] > nonintersecting_probs[i]:
                        nonintersecting_boxes[i] = val
                        nonintersecting_probs[i] = probs[j]
                        break
            if not found:
                nonintersecting_boxes.append(val)
                nonintersecting_probs.append(probs[j])

        for box, prob in zip(nonintersecting_boxes, nonintersecting_probs):
            product_description = {"product" : item, "brand": "", "type": "", "color": "", "other": ""}

            # Blue border for object detected
            object_frame = cv2.rectangle(object_frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), thickness=5) 
            # Blue background for object label
            object_frame = cv2.rectangle(object_frame, (int(box[0]), int(box[3])), (int(box[0]+300), int(box[3])-30) , (255, 0, 0), -1)
            # Object label
            object_frame = cv2.putText(object_frame, item + ": " + format(prob, ".2f")  , (int(box[0]), int(box[3])-5), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 255, 255), 3) 
            cropped_image =  image[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

            # checking if there is an object specific model and setting type using cropped image
            if item in object_specific_models:
                object_specific_model = YOLO(object_specific_models[item])
                object_specific_results = object_specific_model(cropped_image)
                # Black background for classes
                #object_frame = cv2.rectangle(object_frame, (0,0), (250, 30+25*len(np.array(object_specific_results[0].probs.data))) , (0, 0, 0), -1)
                #object_frame = object_specific_results[0].plot(img=object_frame)
                product_description['type'] = " ".join(object_specific_results[0].names[np.argmax(np.array(object_specific_results[0].probs.data.cpu()))].split('_'))
            
            
            # Finding the brand of the product
            brand_results = brand_model(cropped_image)
 
            
            #object_frame = brand_results[0].plot(img=object_frame)
            for brand_index in list(brand_results[0].boxes.cls):

                product_description['brand'] = brand_indices[int(brand_index)]
                logo_box = np.array(brand_results[0].boxes.xyxy.cpu())[0]
                # Green border for logo detected
                object_frame = cv2.rectangle(object_frame, (int(logo_box[0]+box[0]), int(logo_box[1]+box[1])), (int(logo_box[2]+box[0]), int(logo_box[3]+box[1])), (17, 120, 10), thickness=5) 
                # Green background for logo label
                object_frame = cv2.rectangle(object_frame, (int(logo_box[0]+box[0]), int(logo_box[1]+box[1])), (int(logo_box[0]+box[0]+300), int(logo_box[1]+box[1]-30)) , (17, 120, 10), -1)
                # logo label
                object_frame = cv2.putText(object_frame, brand_indices[int(brand_index)] + ": " + format(np.array(brand_results[0].boxes.conf)[0], ".2f")  , (int(logo_box[0]+box[0]), int(logo_box[1]+box[1])-5), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255, 255, 255), 3) 


                logo_image = (get_grayscale(cropped_image[int(logo_box[1]):int(logo_box[3]), int(logo_box[0]):int(logo_box[2])]))
                custom_config = r'--oem 3 --psm 7'
                brand_text = pytesseract.image_to_string(logo_image, config=custom_config)

                print(brand_text)
                brand_text = brand_text.replace("\n", "")
  
                if (not product_description['brand'] == 'Logitech' and nltk.edit_distance(product_description['brand'], brand_text) > 2 and len(brand_text) >= 4 and any(not c.isalnum() for c in brand_text)):
                    #possible the brand has been detected incorrectly
                    product_description['brand'] = brand_text.lower()
                    


                
                break



            # Finding the color of the product
            cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
            reshape = cropped_image_rgb.reshape((cropped_image_rgb.shape[0] * cropped_image_rgb.shape[1], 3))

            cluster = KMeans(n_clusters=5).fit(reshape)

            color = np.array(cluster.cluster_centers_[mode(cluster.labels_)])
            print(color)
            product_description['color'] = min([(np.linalg.norm(np.array(col)-color), name) for (name, col) in rgb_to_text])[1]

            product_description['other'] = image_captioner(cropped_image)
            desc = product_description['other'].split(' ')
            if item in desc and any([(x in desc[:desc.index(item)]) for x in [y[0] for y in rgb_to_text]]):
                product_description['color'] = [x for x in [y[0] for y in rgb_to_text] if x in desc[:desc.index(item)]][0]



            products.append(product_description)


        cv2.imwrite("test.png", object_frame)

        print(products)
        break

