import os
import cv2
import numpy as np
import glob

in_mai = "./image/original/mai_original/"
in_nanami = "./image/original/nanami_original/"
in_nanase = "./image/original/nanase_original/"
out_mai_dir = "./image/face_cut_image/mai_face/"
out_nanami_dir = "./image/face_cut_image/nanami_face/"
out_nanase_dir = "./image/face_cut_image/nanase_face/"


for i in range(3):
    # 読み込みと出力先のディレクトリを選択
    if i == 0:
        s = in_mai
        out_dir = out_mai_dir
    elif i == 1:
        s = in_nanami
        out_dir = out_nanami_dir
    else:
        s = in_nanase
        out_dir = out_nanase_dir

    in_filename = os.listdir(s)

    for n in range(len(in_filename)):
        if str(in_filename[n]) != ".DS_Store":
            image = cv2.imread(s + str(in_filename[n]))

            if image is None:
                print("Not open")
                continue

            image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                            "./cv2/haarcascade_frontalface_alt.xml")
            # 顔認識の実行
            face_list = cascade.detectMultiScale(
                image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

            # 顔が１つ以上検出された時
            if len(face_list) > 0:
                for rect in face_list:
                    x, y, width, height = rect
                    image = image[rect[1]:rect[1] + rect[3],
                                  rect[0]:rect[0] + rect[2]]
                    if image.shape[0] < 64:
                        continue
                    image = cv2.resize(image, (64, 64))

                    print(image.shape)
                    # 保存
                    fileName = os.path.join(out_dir,
                                            str(in_filename[n])+".jpg")
                    cv2.imwrite(str(fileName), image)

            else:   # 顔が検出されなかった時
                print("no face")
                continue
