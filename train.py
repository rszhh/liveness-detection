from livenessnetwork.livenessnet import LivenessNet
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import np_utils
from tensorflow.keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os

import matplotlib
matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-m",
                "--model",
                type=str,
                required=True,
                help="path to trained model")
ap.add_argument("-l",
                "--le",
                type=str,
                required=True,
                help="path to label encoder")
ap.add_argument("-p",
                "--plot",
                type=str,
                default="plot.png",
                help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# 初始化初始学习速率、批大小和要训练的epochs的数量
INIT_LR = 1e-4
BS = 8
EPOCHS = 50

print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

for imagePath in imagePaths:
    # 从文件名中提取类标签，加载图像
    # 将其调整为固定的32x32像素，忽略纵横比
    label = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32))
    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float") / 255.0

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels, 2)

# 将数据划分为训练和测试，使用75%的数据用于训练，剩下的25%用于测试
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels,
                                                  test_size=0.25,
                                                  random_state=42)

# 构造用于数据增强的训练图像生成器
aug = ImageDataGenerator(rotation_range=20,
                         zoom_range=0.15,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.15,
                         horizontal_flip=True,
                         fill_mode="nearest")

# 初始化优化器和模型，向模型打包和传递参数：学习率、衰减等
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = LivenessNet.build(width=32,
                          height=32,
                          depth=3,
                          classes=len(le.classes_))
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# 训练卷积神经网络
print("[INFO] training network for {} epochs...".format(EPOCHS))
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)

print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=BS)
print(
    classification_report(testY.argmax(axis=1),
                          predictions.argmax(axis=1),
                          target_names=le.classes_))

print("[INFO] serializing network to '{}'...".format(args["model"]))
model.save(args["model"], save_format="h5")

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

# 保存历史训练图：训练损失和准确性
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
