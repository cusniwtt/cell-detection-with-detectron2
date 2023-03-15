import albumentations as A
import cv2
import numpy as np
import os

transform = A.Compose([
    A.Resize(width=512, height=512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(limit=45, p=0.5),
    A.CLAHE(p=0.5),
], bbox_params=A.BboxParams(format='yolo', min_area=200, min_visibility=0.1, label_fields=['class_labels']))

def augment(image, bboxes, class_labels):
    transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
    return transformed['image'], transformed['bboxes'], transformed['class_labels']

path = 'datasets/yolov5/annotation/'
file = os.listdir(path)
file = sorted(file)
images = [path + x for x in file if x.endswith('.png')]
txts = [path + x for x in file if x.endswith('.txt')]
save_path = 'datasets/yolov5/train/'
save_images = [save_path + x for x in file if x.endswith('.png')]
save_txts = [save_path + x for x in file if x.endswith('.txt')]

rounds = 10
for i in range(rounds):
    ind = 0
    for img, txt in list(zip(images, txts)):
        print(img)
        image = cv2.imread(img)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annot = np.loadtxt(txt)
        annot_ = []
        class_labels = []
        for row in annot:
            annot_.append([row[1], row[2], row[3], row[4]])
            class_labels.append('nucleus')
        image, annot, label = augment(image, annot_, class_labels)
        save_txt = []
        for item in annot:
            save_txt.append([0, item[0], item[1], item[2], item[3]])

        save_img_path = save_images[ind][:-4] + '_' + str(i) + '.png'
        save_txt_path = save_txts[ind][:-4] + '_' + str(i) + '.txt'
        cv2.imwrite(save_img_path, image)
        print(save_txt)
        np.savetxt(save_txt_path, save_txt, fmt='%d %f %f %f %f')
        ind += 1
        #print('round: ', i, 'image: ', ind)