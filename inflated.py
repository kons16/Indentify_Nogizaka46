import os
import cv2
import glob
from scipy import ndimage

in_mai_dir = "./image/train_data/mai_train/"
in_nanami_dir = "./image/train_data/nanami_train/"
in_nanase_dir = "./image/train_data/nanase_train/"

out_mai_dir = "./image/train_data/mai_infla_train/"
out_nanami_dir = "./image/train_data/nanami_infla_train/"
out_nanase_dir = "./image/train_data/nanase_infla_train/"


for i in range(3):
    if i == 0:
        in_dir = in_mai_dir
        out_dir = out_mai_dir
    elif i == 1:
        in_dir = in_nanami_dir
        out_dir = out_nanami_dir
    else:
        in_dir = in_nanase_dir
        out_dir = out_nanase_dir

    img_file_names = os.listdir(in_dir)

    for i in range(len(img_file_names)):
        if str(img_file_names[i]) != ".DS_Store":
            img = cv2.imread(in_dir + str(img_file_names[i]))

            # 回転
            for ang in [-15, -10, 0, 10, 15]:
                img_rot = ndimage.rotate(img, ang)
                img_rot = cv2.resize(img_rot, (64, 64))
                fileName = os.path.join(
                    out_dir, str(i) + "_" + str(ang) + ".jpg")
                cv2.imwrite(str(fileName), img_rot)

                # 閾値
                img_thr = cv2.threshold(
                    img_rot, 100, 255, cv2.THRESH_TOZERO)[1]
                fileName = os.path.join(out_dir, str(
                    i) + "_" + str(ang) + "thr.jpg")
                cv2.imwrite(str(fileName), img_thr)

                # ぼかし
                img_filter = cv2.GaussianBlur(img_rot, (5, 5), 0)
                fileName = os.path.join(out_dir, str(
                    i) + "_" + str(ang) + "filter.jpg")
                cv2.imwrite(str(fileName), img_filter)
