# liveness-detection

liveness detection with opencv  人脸识别 活体检测

<br>

[中文版README](README_CN.md)

<br>

## Operating Environment

Python 3.8.2 <br>
opencv-python 4.2.0.34 <br>
tensorflow 2.3.0 <br>
Keras 2.3.1 <br>
numpy 1.18.2 <br><br>

## How To Use
<br>

```text
git clone https://github.com/rszhh/liveness-detection.git
```

### 1. Create training datasets

<br>First pick up the phone and take a selfie for about 30 seconds (depending on the number of training data sets you want), walk around and show various micro-expressions, called `real.mp4`；<br><br>

Use the computer camera to shot the phone while the phone is playing the `real.mp4`, or mp4 is shot directly with the computer camera, during which various micro-expressions are displayed and the distance is constantly changed, also in about 30 seconds, and it is called `fake.mp4`；<br><br>

Put the two videos above in `videos/` folder. <br><br>

### 2. Detect and extract ROI from the dataset

<br>Run the command separately：
<br>
```
python gather.py --input videos/fake.mp4 --output dataset/fake --detector face_detector --skip 1
```

```
python gather.py --input videos/real.mov --output dataset/real --detector face_detector --skip 4
```

<br>

### 3. Training in vivo detector

<br>

```
python train.py --dataset dataset --model liveness.model --le le.pickle
```

<br>

### 4. Run

<br>

```
python liveness.py --model liveness.model --le le.pickle --detector face_detector
```

<br><br>
