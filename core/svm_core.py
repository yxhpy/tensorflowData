from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
import numpy as np
import os

# 标准化 StandardScaler


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
print(data)
# 标准化数据
x = data[:, :-1]
y = data[:, -1]
std = StandardScaler()
new_x = std.fit_transform(x)

# 数据集分割
train_x, test_x, train_y, test_y = train_test_split(new_x, y)
print(train_x.shape, train_y.shape)
# 进行训练
svc = SVC()
param_grid = {
    'C': [2 ** i for i in range(-5, 16, 1)],
    # 'C': [16,],
    'gamma': [2 ** i for i in range(-15, 4, 1)],
    # 'gamma': [0.25, ],
    # 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
}
# print(param_grid)
gscv = GridSearchCV(svc, param_grid=param_grid, verbose=10, n_jobs=2)
gscv.fit(train_x, train_y)
# 进行测试
print(gscv.best_params_)
print(gscv.best_score_)
score = gscv.score(test_x, test_y)
print('分数是%f' % score)
