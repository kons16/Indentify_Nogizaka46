import shutil
import glob
import random
import os

in_mai_dir = "./image/face_cut_image/mai_face/"
in_nanami_dir = "./image/face_cut_image/nanami_face/"
in_nanase_dir = "./image/face_cut_image/nanase_face/"

out_train_mai_dir = "./image/train_data/mai_train/"
out_train_nanami_dir = "./image/train_data/nanami_train/"
out_train_nanase_dir = "./image/train_data/nanase_train/"

out_test_mai_dir = "./image/test_data/mai_test/"
out_test_nanami_dir = "./image/test_data/nanami_test/"
out_test_nanase_dir = "./image/test_data/nanase_test/"


for i in range(3):
    if i == 0:
        in_dir = in_mai_dir
        out_train_dir = out_train_mai_dir
        out_test_dir = out_test_mai_dir
    elif i == 1:
        in_dir = in_nanami_dir
        out_train_dir = out_train_nanami_dir
        out_test_dir = out_test_nanami_dir
    else:
        in_dir = in_nanase_dir
        out_train_dir = out_train_nanase_dir
        out_test_dir = out_test_nanase_dir

    img_file_name_list = os.listdir(in_dir)

    for n in range(len(img_file_name_list)):
        if n < 90:
            shutil.copy(in_dir + str(img_file_name_list[n]),
                        out_train_dir + str(img_file_name_list[n]))
        else:
            shutil.copy(in_dir + str(img_file_name_list[n]),
                        out_test_dir + str(img_file_name_list[n]))
