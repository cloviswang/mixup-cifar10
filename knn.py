from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
###
# 聚类精度模板
import numpy as np
from scipy.optimize import linear_sum_assignment


def cluster_acc(y_true, y_pred):
    y_true = np.array(y_true).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    ind = np.asarray(ind)
    ind = np.transpose(ind)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


#
## 朴素贝叶斯与KNN分类
def knn_gnb_lr_lsr(X, labels, title_knn="XXX.KNN", \
                   title_gnb="XXX.GNB", title_lr="XXX.LR", \
                   title_lsr="XXX=LSR", n=3):
    # 划分训练集和测试集
    x_train, x_test, labels_train, labels_test = \
        train_test_split(X, labels, test_size=0.2, random_state=22)

    # 使用KNN进行分类
    knn = KNeighborsClassifier()
    knn.fit(x_train, labels_train)
    label_sample = knn.predict(x_test)
    knn_acc = cluster_acc(labels_test, label_sample)
    print(title_knn, "=", knn_acc)

    # 使用高斯朴素贝叶斯进行分类
    gnb = GaussianNB()  # 使用默认配置初始化朴素贝叶斯
    gnb.fit(x_train, labels_train)  # 训练模型
    label_sample = gnb.predict(x_test)
    gnb_acc = cluster_acc(labels_test, label_sample)
    print(title_gnb, "=", gnb_acc)

    # 线性回归
    lr = LinearRegression()
    lr.fit(x_train, labels_train)
    label_sample = lr.predict(x_test)
    label_sample = np.round(label_sample)
    label_sample = label_sample.astype(np.int64)
    lr_acc = cluster_acc(labels_test, label_sample)
    print(title_lr, "=", lr_acc)

    # Logistic regression 需要事先进行标准化
    # 创建一对多的逻辑回归对象
    # 标准化特征
    scaler = StandardScaler()
    X_ = scaler.fit_transform(X, labels)
    # 划分训练集和测试集
    x_train, x_test, labels_train, labels_test = \
        train_test_split(X_, labels, test_size=0.2)
    log_reg = LogisticRegression(max_iter=3000)  # multinomial
    # 训练模型
    log_reg.fit(x_train, labels_train)
    label_sample = log_reg.predict(x_test)
    lsr_acc = cluster_acc(labels_test, label_sample)
    print(title_lsr, "=", lsr_acc)

    return round(knn_acc, n), round(gnb_acc, n), round(lr_acc, n), round(lsr_acc, n)


def get_imgdata(file, sfile, re_size=16, n=5):
    import pickle
    import numpy as np
    from skimage.transform import resize

    def unpickle(file):
        with open(file, 'rb') as f:
            cifar_dict = pickle.load(f, encoding='latin1')
        return cifar_dict

    # 定义用来存放图像数据 图像标签 图像名称list  最后返回的cifar_image cifar_label即是图像cifar-10 对应的数据和标签
    tem_cifar_image = []
    tem_cifar_label = []
    tem_cifar_image_name = []
    for i in range(1, n + 1):
        # 存放是你的文件对应的目录
        cifar_file = sfile + str(i)
        cifar = unpickle(cifar_file)
        cifar_label = cifar['labels']
        cifar_image = cifar['data']
        cifar_image_name = cifar['filenames']
        # 使用transpose()函数是因为cifar存放的是图像标准是 通道数 高 宽 所以要修改成  高 宽 通道数
        cifar_image = cifar_image.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        cifar_image = np.asarray([resize(x_img, [re_size, re_size]) for x_img in cifar_image])
        cifar_label = np.array(cifar_label)
        cifar_image_name = np.array(cifar_image_name)
        tem_cifar_image.append(cifar_image)
        tem_cifar_label.append(cifar_label)
        tem_cifar_image_name.append(cifar_image_name)
    cifar_image = np.concatenate(tem_cifar_image)
    cifar_label = np.concatenate(tem_cifar_label)
    cifar_image_name = np.concatenate(tem_cifar_image_name)
    return cifar_image, cifar_label, cifar_image_name


file = "D:\github\mixup-cifar10\data\cifar-10-batches-py\\batches.meta"
sfile = "D:\github\mixup-cifar10\data\cifar-10-batches-py\\data_batch_"
n = 5  # n表示要获取几个数据集
re_size = 8
X = []
Y = []
z = []
X, Y, Z = get_imgdata(file, sfile, re_size, n)
X = X.reshape(n * 10000, -1)
X_ = []
Y_ = []
for i in range(n):
    X_.append(X[i * 10000:(i + 1) * 10000])
    Y_.append(Y[i * 10000:(i + 1) * 10000])
X_ = np.array(X_)
Y_ = np.array(Y_)
knn_acc = []
gnb_acc = []
lr_acc = []
lsr_acc = []
for i in range(n):
    t1, t2, t3, t4 = \
        knn_gnb_lr_lsr(X_[i], Y_[i])
    knn_acc.append(t1)
    gnb_acc.append(t2)
    lr_acc.append(t3)
    lsr_acc.append(t4)
# 使用pandas输出
title1 = []
for i in range(n):
    t = 'data_bath' + str(i + 1)
    title1.append(t)
title2 = ["KNN     ", "Naive Bayes", "linear regression", "Logistic regression"]
data = pd.DataFrame([knn_acc, gnb_acc, lr_acc, lsr_acc], index=title2, columns=title1)
print(data)
