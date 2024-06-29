# import các package cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import legacy
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

# Khởi tạo learning rate, số epoch và batch size
INIT_LR = 1e-4
EPOCHS = 20
BS = 16

DIRECTORY = r"C:/computervison/face_mask_rcn/dataset"
CATEGORIES = ["with_mask", "without_mask"]

# Load ảnh và nhãn từ thư mục
print("[INFO] loading images...")

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# one-hot encoding nhãn
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32")
labels = np.array(labels)

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

# Tạo data augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

# Load mô hình VGG16
baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Tạo head của mô hình để đặt trên mô hình cơ sở
headModel = baseModel.output
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# Đặt head FC model lên trên mô hình cơ sở để tạo thành mô hình hoàn chỉnh
model = Model(inputs=baseModel.input, outputs=headModel)

# Đóng băng tất cả các layer trong base model để chúng không được cập nhật trong quá trình huấn luyện
for layer in baseModel.layers:
    layer.trainable = False

# Compile mô hình
print("[INFO] compiling model...")
opt = legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train mô hình
print("[INFO] training head...")
H = model.fit(
    aug.flow(trainX, trainY, batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX, testY),
    validation_steps=len(testX) // BS,
    epochs=EPOCHS)

# Đánh giá mô hình trên tập kiểm tra
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# Tìm nhãn dự đoán cho các ảnh trong tập kiểm tra dựa trên xác suất dự đoán của mô hình
predIdxs = np.argmax(predIdxs, axis=1)

# In ra báo cáo đánh giá mô hình
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

# Lưu mô hình xuống đĩa
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")