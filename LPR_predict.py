import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Dropout, Flatten, Dense, \
    GlobalAveragePooling2D
from tensorflow.keras import Model

# 屏蔽tensorflow中的warning信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = "./test"     # 测试文件路径
checkpoint_save_path = "./checkpoint_good/LPR.ckpt"

dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
        10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K',
        20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T', 28:'U', 29:'V',
        30:'W', 31:'X', 32:'Y', 33:'Z', 34:'冀', 35:'新', 36:'鄂', 37:'宁', 38:'桂', 39:'黑',
        40:'湘', 41:'皖', 42:'云', 43:'豫', 44:'蒙', 45:'赣', 46:'吉', 47:'辽', 48:'苏', 49:'甘',
        50:'晋', 51:'浙', 52:'闽', 53:'渝', 54:'贵', 55:'陕', 56:'粤', 57:'川', 58:'鲁', 59:'琼',
        60:'青', 61:'藏', 62:'京', 63:'津', 64:'沪'
}



class ConvBNRelu(Model):
    def __init__(self, ch, kernelsz=3, strides=1, padding='same'):
        super(ConvBNRelu, self).__init__()
        self.model = tf.keras.models.Sequential([
            Conv2D(ch, kernelsz, strides=strides, padding=padding),
            BatchNormalization(),
            Activation('relu')
        ])

    def call(self, x):
        x = self.model(x, training=False) #在training=False时，BN通过整个训练集计算均值、方差去做批归一化，training=True时，通过当前batch的均值、方差去做批归一化。推理时 training=False效果好
        return x


class InceptionBlk(Model):
    def __init__(self, ch, strides=1):
        super(InceptionBlk, self).__init__()
        self.ch = ch
        self.strides = strides
        self.c1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c2_2 = ConvBNRelu(ch, kernelsz=3, strides=1)
        self.c3_1 = ConvBNRelu(ch, kernelsz=1, strides=strides)
        self.c3_2 = ConvBNRelu(ch, kernelsz=5, strides=1)
        self.p4_1 = MaxPool2D(3, strides=1, padding='same')
        self.c4_2 = ConvBNRelu(ch, kernelsz=1, strides=strides)

    def call(self, x):
        x1 = self.c1(x)
        x2_1 = self.c2_1(x)
        x2_2 = self.c2_2(x2_1)
        x3_1 = self.c3_1(x)
        x3_2 = self.c3_2(x3_1)
        x4_1 = self.p4_1(x)
        x4_2 = self.c4_2(x4_1)
        # concat along axis=channel
        x = tf.concat([x1, x2_2, x3_2, x4_2], axis=3)
        return x


class Inception10(Model):
    def __init__(self, num_blocks, num_classes, init_ch=16, **kwargs):
        super(Inception10, self).__init__(**kwargs)
        self.in_channels = init_ch
        self.out_channels = init_ch
        self.num_blocks = num_blocks
        self.init_ch = init_ch
        self.c1 = ConvBNRelu(init_ch)
        self.blocks = tf.keras.models.Sequential()
        for block_id in range(num_blocks):
            for layer_id in range(2):
                if layer_id == 0:
                    block = InceptionBlk(self.out_channels, strides=2)
                else:
                    block = InceptionBlk(self.out_channels, strides=1)
                self.blocks.add(block)
            # enlarger out_channels per block
            self.out_channels *= 2
        self.p1 = GlobalAveragePooling2D()
        self.f1 = Dense(num_classes, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.blocks(x)
        x = self.p1(x)
        y = self.f1(x)
        return y


def main():
    model = Inception10(num_blocks=2, num_classes=65)

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    else:
        print("No model exist!")
        exit(0)

    ac_conut = 0
    sum = 0
    for file in os.listdir(path):
        pathd = path + '/' + file

        for filed in os.listdir(pathd):
            img_path = pathd + '/' + filed
            img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
            # img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=0)
            img = cv.resize(img, (46, 90))

            img_arr = img / 255.0
            x_predict = img_arr[tf.newaxis, ...]
            result = model.predict(x_predict)
            result = list(result[0])
            index = result.index(max(result))
            print(img_path, " prdect: ", dict[index])
            if dict[index] == file[0]:
                ac_conut += 1
            sum += 1

    print("Accuracy:", ac_conut / sum)

if __name__ == '__main__':
    main()







