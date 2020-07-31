import numpy as np
import tensorflow as tf
import os
import cv2 as cv
from LPR_CNN import Inception10
from License_plate_detection_and_cut import detction_and_cut,show
# 屏蔽tensorflow中的warning信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

path = "./test"     # 测试文件路径
img_path = './test_picture/test7.jpg'
checkpoint_save_path = "./checkpoint_good/LPR.ckpt"

dict = {0:'0',1:'1',2:'2',3:'3',4:'4',5:'5',6:'6',7:'7',8:'8',9:'9',
        10:'A', 11:'B', 12:'C', 13:'D', 14:'E', 15:'F', 16:'G', 17:'H', 18:'J', 19:'K',
        20:'L', 21:'M', 22:'N', 23:'P', 24:'Q', 25:'R', 26:'S', 27:'T', 28:'U', 29:'V',
        30:'W', 31:'X', 32:'Y', 33:'Z', 34:'冀', 35:'新', 36:'鄂', 37:'宁', 38:'桂', 39:'黑',
        40:'湘', 41:'皖', 42:'云', 43:'豫', 44:'蒙', 45:'赣', 46:'吉', 47:'辽', 48:'苏', 49:'甘',
        50:'晋', 51:'浙', 52:'闽', 53:'渝', 54:'贵', 55:'陕', 56:'粤', 57:'川', 58:'鲁', 59:'琼',
        60:'青', 61:'藏', 62:'京', 63:'津', 64:'沪'}

def load_model():
    model = Inception10(num_blocks=2, num_classes=65)

    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)
    else:
        print("No model exist!")
        exit(0)
    return model

def test():
    model = load_model()

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
            if dict[index] == filed[0]:
                ac_conut += 1
            sum += 1

    print("Accuracy:", ac_conut / sum)

def predict(img_path):
    img = detction_and_cut(img_path)
    model = load_model()
    str=''
    for i in range(1,8):
        img_character = cv.cvtColor(img[i], cv.COLOR_GRAY2RGB)
        img_character = img_character / 255.0
        try:
            x_predict = img_character[tf.newaxis, ...]
            result = model.predict(x_predict)
        except:
            print("error!")
            return 0
        result = list(result[0])
        index = result.index(max(result))
        str += dict[index]
    print("Result predict is : ",str)
    print(img[0])
    show("Image",img[0])
    cv.waitKey(0)
    cv.destroyAllWindows()
    return 0




if __name__ == '__main__':
    # test()
    predict(img_path)







