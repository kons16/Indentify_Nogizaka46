''' Fine tuning を利用する '''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.models import Model
from keras.models import Sequential
from keras.utils import np_utils
from keras import models, optimizers
from keras.optimizers import SGD

# 訓練データの作成
x_train = []
y_train = []
mai_train = os.listdir("./image/train_data/mai_infla_train/")
nanami_train = os.listdir("./image/train_data/nanami_infla_train/")
nanase_train = os.listdir("./image/train_data/nanase_infla_train/")
for name_i in range(3):
    if name_i == 0:
        img_file_names = mai_train
    elif name_i == 1:
        img_file_names = nanami_train
    else:
        img_file_names = nanase_train

    for i in range(len(img_file_names)):
        if img_file_names[i] != ".DS_Store":
            if name_i == 0:
                img = cv2.imread("./image/train_data/mai_infla_train/" +
                                 str(img_file_names[i]))
            elif name_i == 1:
                img = cv2.imread("./image/train_data/nanami_infla_train/" +
                                 str(img_file_names[i]))
            else:
                img = cv2.imread("./image/train_data/nanase_infla_train/" +
                                 str(img_file_names[i]))

            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            x_train.append(img)
            y_train.append(name_i)


# テストデータの作成
x_test = []
y_test = []
mai_test = os.listdir("./image/test_data/mai_test/")
nanami_test = os.listdir("./image/test_data/nanami_test/")
nanase_test = os.listdir("./image/test_data/nanase_test/")
for name_i in range(3):
    if name_i == 0:
        img_file_names = mai_test
    elif name_i == 1:
        img_file_names = nanami_test
    else:
        img_file_names = nanase_test

    for i in range(len(img_file_names)):
        if img_file_names[i] != ".DS_Store":
            if name_i == 0:
                img = cv2.imread("./image/test_data/mai_test/" +
                                 str(img_file_names[i]))
            elif name_i == 1:
                img = cv2.imread("./image/test_data/nanami_test/" +
                                 str(img_file_names[i]))
            else:
                img = cv2.imread("./image/test_data/nanase_test/" +
                                 str(img_file_names[i]))

            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            x_test.append(img)
            y_test.append(name_i)


x_train = np.array(x_train)
x_test = np.array(x_test)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)


# VGG16モデルと学習済み重みをロード
vgg_conv = VGG16(weights='imagenet',
                 include_top=False, input_shape=(64, 64, 3))

for layer in vgg_conv.layers[:-4]:
    layer.trainable = False

model = models.Sequential()
model.add(vgg_conv)
model.add(Flatten())
model.add(Dense(256, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# 学習
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=50,
                    validation_split=0.1,
                    verbose=1)

score = model.evaluate(x_test, y_test, verbose=0)
print('test accuracy : ', score[1])

model.save("nogi_finetuning.h5")


# acc, val_accのプロット
plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_acc"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
save_dir = "./kekka"
plt.savefig(os.path.join(save_dir, "graph_finetuning.png"))
