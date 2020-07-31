import tensorflow as tf
import cv2 as cv
import numpy as np
import os

path = './character_data_augmentation2'
savepath1 = './LPR_train_data.npy'
savepath2 = './LPR_train_label.npy'
savepath = './LPR_data.npy'

dict = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,
        'A':10, 'B':11, 'C':12, 'D':13, 'E':14, 'F':15, 'G':16, 'H':17, 'J':18, 'K':19,
        'L':20, 'M':21, 'N':22, 'P':23, 'Q':24, 'R':25, 'S':26, 'T':27, 'U':28, 'V':29,
        'W':30, 'X':31, 'Y':32, 'Z':33, '冀':34, '新':35, '鄂':36, '宁':37, '桂':38, '黑':39,
        '湘':40, '皖':41, '云':42, '豫':43, '蒙':44, '赣':45, '吉':46, '辽':47, '苏':48, '甘':49,
        '晋':50, '浙':51, '闽':52, '渝':53, '贵':54, '陕':55, '粤':56, '川':57, '鲁':58, '琼':59,
        '青':60, '藏':61, '京':62, '津':63, '沪':64
}

def creat_data_npy(path):

    data, x, y_ = [], [], []

    for file in os.listdir(path):
        pathd = path + '/' + file
        for filed in os.listdir(pathd):
            img_path = pathd + '/' + filed
            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
            img = cv.resize(img,(46,90))
            img = np.array(img/255.0)
            x.append(img)
            y_.append(dict[file])
        print('loading: '+file)

    # 随机打乱数据（因为原始数据是顺序的，顺序不打乱会影响准确率）
    # seed: 随机数种子，是一个整数，当设置之后，每次生成的随机数都一样（为方便教学，以保每位同学结果一致）
    np.random.seed(116)  # 使用相同的seed，保证输入特征和标签一一对应
    np.random.shuffle(x)
    np.random.seed(116)
    np.random.shuffle(y_)

    num = len(y_)

    # 将打乱后的数据集分割为训练集和测试集，训练集为前120行，测试集为后30行
    x_train = x[:-int(num*0.3)]
    y_train = y_[:-int(num*0.3)]
    x_test = x[-int(num*0.3):]
    y_test = y_[-int(num*0.3):]

    # 转换x的数据类型，否则后面矩阵相乘时会因数据类型不一致报错
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = np.reshape(y_train,(len(y_train) ,-1))
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_test = np.reshape(y_test, (len(y_test) ,-1))
    return np.asarray([x_train,y_train,x_test,y_test])

if __name__ == '__main__':
    data = creat_data_npy(path)
    # np.save(savepath1, data)
    # np.save(savepath2, label)
    np.save(savepath,data)
    print("Over")