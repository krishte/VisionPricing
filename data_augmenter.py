import albumentations as A
import cv2
from os import listdir
from os.path import isfile, join

i=0
transform = A.Compose([
        A.RandomScale(p=0.7),
        A.Rotate(limit=(-10, 10), p=0.8),
        A.BBoxSafeRandomCrop(p=1.0),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10,p=0.8),
        A.ISONoise(),
        A.RandomBrightnessContrast(p=0.3)
        
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

path = "datasets/brand_dataset/labels/train"
onlyfiles = [f for f in listdir(path)]
images = [f for f in listdir("datasets/brand_dataset/images/train")]
images_without_filetype = [f.split('.')[0] for f in images]
for filename in onlyfiles:
    i += 1
    print(i)
    with open('{}'.format(path + "/" + filename), 'r') as file:
        data = file.read()
        lines = data.split('\n')
        boxes = [[max(float(x), 0.000001) for x in line.split(' ')[1:]] for line in lines]
        classes = [int(line.split(' ')[0]) for line in lines]
        image = cv2.imread("datasets/brand_dataset/images/train/" + images[images_without_filetype.index(filename.split('.')[0])])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(boxes)

        transformed = transform(image=image, bboxes=boxes, class_labels=classes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']
        transformed_class_labels = transformed['class_labels']

        new_lines = []
        for box, class_label in zip(transformed_bboxes, transformed_class_labels):
            temp = [class_label] + list(["{:.6f}".format(x) for x in box])
            temp = " ".join(str(x) for x in temp)
            new_lines.append(temp)

        new_data = '\n'.join(new_lines)
        with open('{}'.format("datasets/brand_dataset_augmented/labels/train/augmented_" + filename), 'w+') as file:
            file.write(new_data)
        cv2.imwrite("datasets/brand_dataset_augmented/images/train/augmented_" + filename.split('.')[0] + ".png", transformed_image)
