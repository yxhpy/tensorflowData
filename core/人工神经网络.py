import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import numpy as np
import os

mnist = tf.keras.datasets.mnist
# 读入数据
data_name = '../data/krkopt.data'
data = np.genfromtxt(data_name, delimiter=',', dtype=str)

# 讲数母转化为数字
for i in range(ord('a'), ord('h') + 1):
    data[data == chr(i)] = i

# 将标签转化为数字
classfication = set(data[:, -1])
for tag, i in enumerate(classfication):
    data[data == i] = tag
data = np.array(data, dtype=float)
data = data[data[:, -1] < 2,]
# 标准化数据
x = data[:, :-1]
y = data[:, -1]
std = StandardScaler()
new_x = std.fit_transform(x)
# 数据集分割
x_train, x_test, y_train, y_test = train_test_split(new_x, y)
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(6,)),
    tf.keras.layers.Dense(1024, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test, verbose=2)
