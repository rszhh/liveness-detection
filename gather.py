import numpy as np
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i",
                "--input",
                type=str,
                required=True,
                help="path to input video")
ap.add_argument("-o",
                "--output",
                type=str,
                required=True,
                help="path to output directory of cropped faces")
ap.add_argument("-d",
                "--detector",
                type=str,
                required=True,
                help="path to OpenCV's deep learning face detector")
ap.add_argument("-c",
                "--confidence",
                type=float,
                default=0.5,
                help="minimum probability to filter weak detections")
ap.add_argument("-s",
                "--skip",
                type=int,
                default=16,
                help="# of frames to skip before applying face detection")
args = vars(ap.parse_args())

print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join(
    [args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

vs = cv2.VideoCapture(args["input"])
read = 0
saved = 0

while True:
    (grabbed, frame) = vs.read()
    # 如果已经frame没有被grabbed，就到达了视频流的最后
    if not grabbed:
        break

    # 增加到目前为止读取的帧总数
    read += 1
    if read % args["skip"] != 0:
        continue

    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # 将blob通过网络传递，得到检测、预测结果
    net.setInput(blob)
    detections = net.forward()

    if len(detections) > 0:
        # 找到概率最大的边界框
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > args["confidence"]:
            # 计算面部边界框(x, y)坐标并提取面部ROI
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face = frame[startY:endY, startX:endX]

            p = os.path.sep.join([args["output"], "{}.png".format(saved)])
            cv2.imwrite(p, face)
            saved += 1
            print("[INFO] saved {} to disk".format(p))

vs.release()
cv2.destroyAllWindows()
