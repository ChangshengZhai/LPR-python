import cv2 as cv
import numpy as np
import os

img_path = "E:/机器学习/较复杂环境下车牌号的识别/test_picture/test.jpg"
save_path = "E:/机器学习/较复杂环境下车牌号的识别/test/test_save"



def separate_color_blue(img):       # HSV阈值难以确定，暂时不用
    #颜色提取
    # gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv = np.array([180, 43, 0])  # 提取颜色的低值
    high_hsv = np.array([300, 180,155])  # 提取颜色的高值
    mask = cv.inRange(img, lowerb=lower_hsv, upperb=high_hsv)
    print("颜色提取完成")
    return mask

def separate_color_blue2(image):
    #颜色提取
    for i in range (len(image)):
        for j in range(len(image[i])):
            if image[i][j][2]<15 and image[i][j][1]<127 and image[i][j][1]>15 and image[i][j][0]>60:
               image[i][j] = [255, 255, 255]
               # 填补部分区域
               # if j+5<= len(image[i]) and i+5<len(image):
               #     cv.line(image, (i, j-5),(i,j+5), color=(255, 255, 255),thickness=10,lineType=4)
               # j = j + 5
               # continue
            elif image[i][j][2]<70 and image[i][j][1]<127 and image[i][j][0]>100:
                image[i][j] = [255,255,255]
            elif image[i][j][2]>100 and image[i][j][1]>100:
                image[i][j] = [0,0,0]
            else:
                image[i][j] = [0,0,0]

    # print("颜色提取完成")
    return image

def binary(img):
    #二值化处理去燥
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j]< 130:
                img[i][j]=0
            else:
                img[i][j] = 255
    return img

def show(name, img):
    #显示图片
    cv.namedWindow(str(name), cv.WINDOW_AUTOSIZE)
    cv.imshow(str(name), img)



def contour(img1,img2):
    #检测轮廓
    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    ret, img1 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)

    image, contours, hier = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    for c in contours:  # 遍历轮廓
        rect = cv.minAreaRect(c)  # 生成最小外接矩形
        box_ = cv.boxPoints(rect)
        h = abs(box_[3, 1] - box_[1, 1])
        w = abs(box_[3, 0] - box_[1, 0])
        # print("宽，高", w, h)
        # 只保留需要的轮廓
        if (h > 1400 or w > 1400):
            continue
        if (h < 20 or w < 10):
            continue
        if (w / h > 6 or w / h < 2):
            continue
        count += 1
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        angle = rect[2]  # 获取矩形相对于水平面的角度
        # print("angle", angle)
        # print("坐标", box)
        if angle > 0:
            if abs(angle) > 45:
                angle = 90 - abs(angle)
        else:
            if abs(angle) > 45:
                angle = (90 - abs(angle))
        # 绘制矩形
        cv.drawContours(img2, [box], 0, (0, 0, 255), 2)

    # print("轮廓数量", count)
    return img1,img2, box, angle


def rotate(img, angle):
    #旋转图片
    (h, w) = img.shape[:2]  #获得图片高，宽
    center = (w // 2, h // 2)  #获得图片中心点
    img_ratete = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, img_ratete, (w, h))
    return rotated

# 裁剪车牌
def cut1(img, box):
    #从轮廓出裁剪图片
    x, y=[], []
    for i in range(len(box)):
        x.append(box[i][0])
        y.append(box[i][1])
    x1, y1 = min(x), min(y)  #获取左上角坐标
    x2, y2 = max(x), max(y)  #获取右下角坐标
    img_cut = img[y1:y2, x1:x2]  #切片裁剪图像
    return img_cut

# 裁剪出字符
def cut2(img_cut):
    img_cut = cv.resize(img_cut, (440, 140))
    img1 = img_cut[25:115, 15:61]
    img2 = img_cut[25:115, 72:118]
    img3 = img_cut[25:115, 151:197]
    img4 = img_cut[25:115, 208:254]
    img5 = img_cut[25:115, 265:311]
    img6 = img_cut[25:115, 322:368]
    img7 = img_cut[25:115, 379:425]
    return img1,img2,img3,img4,img5,img6,img7


def cut_test_save():
    bool = os.path.exists(save_path)
    if bool == False:
        os.makedirs(save_path)
    # 解决imread不能读取中文路径
    # img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.COLOR_BGR2HSV)
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
    w = len(img[0])
    h = len(img)
    print(w)
    print(h)
    img = cv.resize(img, (int(w / 2), int(h / 2)))
    print("begin")
    image = img
    show("img", img)

    img_separate = separate_color_blue2(image)  # 提取蓝色框先
    cv.imencode('.png', img_separate)[1].tofile(save_path + '/' + 'test.jpg')
    show("img_separate", img_separate)

    img_contours2, img2, box, angle = contour(img_separate, img)  # 轮廓检测，获取最外层矩形框的偏转角度
    show("img2", img2)
    show("img_contours2", img_contours2)
    print("坐标2", box)
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=0)
    img = cv.resize(img, (int(w / 2), int(h / 2)))
    img = binary(img)
    img_cut = cut1(img, box)
    img_cut_rotate = rotate(img_cut, angle)
    img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7 = cut2(img_cut_rotate)
    show("img_cut_rotate", img_cut_rotate)
    cv.imencode('.png', img_cut_rotate)[1].tofile(save_path + '/' + 'img_cut_rotate.png')
    cv.imencode('.png', img_cut)[1].tofile(save_path + '/' + 'img_cut.png')
    cv.imencode('.png', img_cut1)[1].tofile(save_path + '/' + 'img_cut1.png')
    cv.imencode('.png', img_cut2)[1].tofile(save_path + '/' + 'img_cut2.png')
    cv.imencode('.png', img_cut3)[1].tofile(save_path + '/' + 'img_cut3.png')
    cv.imencode('.png', img_cut4)[1].tofile(save_path + '/' + 'img_cut4.png')
    cv.imencode('.png', img_cut5)[1].tofile(save_path + '/' + 'img_cut5.png')
    cv.imencode('.png', img_cut6)[1].tofile(save_path + '/' + 'img_cut6.png')
    cv.imencode('.png', img_cut7)[1].tofile(save_path + '/' + 'img_cut7.png')
    show("img_cut", img_cut)
    show("img_cut1", img_cut1)
    show("img_cut2", img_cut2)
    show("img_cut3", img_cut3)
    show("img_cut4", img_cut4)
    show("img_cut5", img_cut5)
    show("img_cut6", img_cut6)
    show("img_cut7", img_cut7)

    cv.waitKey(0)
    cv.destroyAllWindows()

def detction_and_cut(path):
    try:
        img = cv.imdecode(np.fromfile(path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
    except ValueError:
        print("图片解析失败！")
        return
    w, h = len(img[0]), len(img)
    img = cv.resize(img, (w // 2, h // 2))
    img2 = img
    img_separate = separate_color_blue2(img)  # 提取蓝色框先
    try:
        img_contours, img2, box, angle = contour(img_separate, img2)  # 轮廓检测，获取最外层矩形框的偏转角度
    except ValueError:
        print("未检测到车牌！")
        return
    img = cv.imdecode(np.fromfile(path, dtype=np.uint8), flags=0)
    img = cv.resize(img, (w // 2, h // 2))
    img = binary(img)
    img_cut = cut1(img, box)
    img_cut_rotate = rotate(img_cut, angle)
    img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7 = cut2(img_cut_rotate)
    return [img_cut_rotate, img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7]

if __name__ == "__main__":
    cut_test_save()


