import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import csv

# ラベル定義
dic_label = {}
with open("labels.txt") as f:
    reader = csv.reader(f, delimiter=" ")
    for row in reader:
        dic_label[int(row[0])] = row[1]


model = load_model("keras_model.h5")
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
image = Image.open("test.jpg")
size = (224, 224)
image = ImageOps.fit(image, size, Image.ANTIALIAS)

image_array = np.asarray(image)
normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
data[0] = normalized_image_array

predictions = model.predict(data)   # リストのリストになっている
for i, pred in enumerate(predictions[0]):
    print(f"{dic_label[i]}の確率＝{pred}")

arg = np.argmax(predictions[0])
print("ということで")
print(f"{dic_label[arg]}")

pass