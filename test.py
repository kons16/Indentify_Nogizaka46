import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import load_model
import sys


def detect_face(image):
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        "./cv2/haarcascade_frontalface_alt.xml")

    face_list = cascade.detectMultiScale(
        image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

    if len(face_list) > 0:
        for rect in face_list:
            x, y, width, height = rect
            cv2.rectangle(image, tuple(rect[0:2]), tuple(
                rect[0:2] + rect[2:4]), (255, 0, 0), thickness=3)
            img = image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
            if image.shape[0] < 64:
                print("too small")
                continue
            img = cv2.resize(image, (64, 64))
            img = np.expand_dims(img, axis=0)
            name = detect_who(img)
            print(name)
            cv2.putText(image, name, (x, y + height + 20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    else:
        print("no face")
    return image


def detect_who(img):
    name = ""
    print(model.predict(img))
    nameNumLabel = np.argmax(model.predict(img))
    if nameNumLabel == 0:
        name = "Shiraishi Mai"
    elif nameNumLabel == 1:
        name = "Hashimoto Nanami"
    elif nameNumLabel == 2:
        name = "Nishino Nanase"

    return name


model = load_model('./nogi.h5')

origin_set = os.listdir("./image/test_data/original_test/")
for i in range(len(origin_set)):
    if str(origin_set[i]) != ".DS_Store":
        image = cv2.imread("./image/test_data/original_test/" +
                           str(origin_set[i]))
        b, g, r = cv2.split(image)
        image = cv2.merge([r, g, b])
        whoImage = detect_face(image)

        plt.imshow(whoImage)
        save_dir = "./kekka"
        plt.savefig(os.path.join(save_dir, "kekka{}.png".format(i)))
