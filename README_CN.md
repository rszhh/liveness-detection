# 活体检测

使用opencv建立基于人脸识别的活体检测技术 <br><br>

## 运行环境

Python 3.8.2 <br>
opencv-python 4.2.0.34 <br>
tensorflow 2.3.0 <br>
Keras 2.3.1 <br>
numpy 1.18.2 <br><br>

## 怎样使用
<br>

```text
git clone https://github.com/rszhh/liveness-detection.git
```


### 1. 制作训练数据集

<br>首先拿起手机进行自拍，时间大约30秒左右（取决于你想要的训练数据集数量），来回走动并展示各种微表情，命名为`real.mp4`；<br><br>

然后使用电脑摄像头拍摄`real.mp4`，或者直接使用电脑摄像头拍摄视频，拍摄过程中也展示各种微表情并不断改变距离，时间也为30秒左右，命名为`fake.mp4`；<br><br>

将上面两个视频放入`videos/`文件夹内。<br><br>

### 2. 从数据集中检测并提取ROI

<br>分别运行命令：
<br>
```
python gather.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 1
```

```
python gather.py --input videos/real.mov --output dataset/real --detector face_detector --skip 4
```

<br>

### 3. 训练活体检测器

<br>

```
python train.py --dataset dataset --model liveness.model --le le.pickle
```

<br>

### 4. 运行

<br>

```
python liveness.py --model liveness.model --le le.pickle --detector face_detector
```

<br><br>
