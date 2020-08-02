import cv2 as cv
import numpy as np
import os

img_path = "E:/机器学习/较复杂环境下车牌号的识别/LPR/test_picture/test5.jpg"
save_path = "E:/机器学习/较复杂环境下车牌号的识别/test/test_save"

def show(name, img):
    #显示图片
    cv.namedWindow(str(name), cv.WINDOW_AUTOSIZE)
    cv.imshow(str(name), img)

def binary(img):
    #二值化处理去燥
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j]< 130:
                img[i][j]=0
            else:
                img[i][j] = 255
    return img

def separate_color_blue(img):       # HSV阈值难以确定，暂时不用
    #颜色提取
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    lower_hsv = np.array([105, 110, 115])  # 提取颜色的低值
    high_hsv = np.array([130, 255,255])  # 提取颜色的高值
    mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    mask = binary(mask)
    # print("颜色提取完成")
    return mask


def contour(img1,img2):
    #检测轮廓
    # img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # show("kkk",img1)
    # cv.waitKey(0)
    ret, img1 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)

    image, contours, hier = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv.minAreaRect(c)  # 生成最小外接矩形
        h = min([int(rect[1][0]), int(rect[1][1])])
        w = max([int(rect[1][0]), int(rect[1][1])])
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        cv.drawContours(img1, [box], 0, (255, 255, 255), 8)

    # 补充完矩形框后，再检测一次，寻找车牌目标矩形
    image, contours, hier = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    max_w = 0
    max_h = 0
    max_angle = 0
    for c in contours:  # 遍历轮廓
        approx = cv.approxPolyDP(c, epsilon=5,closed=True)
        rect = cv.minAreaRect(approx)  # 生成最小外接矩形
        h = min([int(rect[1][0]), int(rect[1][1])])
        w = max([int(rect[1][0]), int(rect[1][1])])
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        cv.drawContours(img2, [box], 0, (0, 0, 255), 2)

        # 只保留需要的轮廓
        if (h > 1000 or w > 1000):
            continue
        if (h < 20 or w < 10):
            continue
        if (w / h > 5 or w / h <= 1.5):
            continue
        count += 1
        angle = rect[2]  # 获取矩形相对于水平面的角度
        if angle > 0:
            if abs(angle) > 45:
                angle = 90 - abs(angle)
        else:
            if abs(angle) > 45:
                angle = (90 - abs(angle))
        # 筛选出最大的矩形对应的坐标
        if w > max_w and h > max_h:
            max_w, max_h = w, h
            max_box = box.copy()
            max_angle = angle
        # print("angle", angle)
        # print("坐标", box)

    # show("img_1",img1)
    # # cv.drawContours(img2, contours, -1, color=(0,0,255),thickness=2)
    # show("img1_contours",img2)
    # print("轮廓数量", count)
    # cv.waitKey()
    # print("max_box",max_box)
    # if flag == False:
    #     cv.waitKey()
    # show("max_box",img2)
    if count==0:
        print("Error, can not find License Plate!")
        exit()
    cv.drawContours(img2, [max_box], 0, (0, 255, 0), 10)

    return img1,img2, max_box, max_angle


def rotate(img, angle):
    #旋转图片
    (h, w) = img.shape[:2]  #获得图片高，宽
    center = (w // 2, h // 2)  #获得图片中心点
    img_ratete = cv.getRotationMatrix2D(center, angle, 1)
    rotated = cv.warpAffine(img, img_ratete, (w, h))
    return rotated

# 裁剪车牌
def cut1(img, box,flag):
    #从轮廓出裁剪图片
    x, y=[], []
    for i in range(len(box)):
        x.append(box[i][0])
        y.append(box[i][1])
    x1, y1 = min(x), min(y)  #获取左上角坐标
    x2, y2 = max(x), max(y)  #获取右下角坐标
    x1, y1 = max([0,x1]), max([0,y1])
    x2, y2 = max([0, x2]), max([0, y2])
    # p为校验值
    p = 0
    if flag == False:
        p = int(len(img) * 0.05)
    img_cut = img[y1:y2, x1 + p:x2 - 2*p,:]  #切片裁剪图像
    return img_cut

# 裁剪出字符
def cut2(img_cut):
    img_cut = cv.resize(img_cut, (440, 140))
    img_cut = binary(cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY))
    img1 = img_cut[15:125, 15:61]
    img2 = img_cut[15:125, 72:118]
    img3 = img_cut[15:125, 151:197]
    img4 = img_cut[15:125, 208:254]
    img5 = img_cut[15:125, 265:311]
    img6 = img_cut[15:125, 322:368]
    img7 = img_cut[15:125, 379:425]
    return img1,img2,img3,img4,img5,img6,img7


def cut_test_save():
    bool = os.path.exists(save_path)
    if bool == False:
        os.makedirs(save_path)
    # 解决imread不能读取中文路径
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
    img = cv.resize(img,(512,512))
    img_separate = separate_color_blue(img.copy())  # 提取蓝色框先
    cv.imencode('.png', img_separate)[1].tofile(save_path + '/' + 'test.jpg')
    show("img_separate", img_separate)

    img_contours2, img2, box, angle = contour(img_separate, img.copy())  # 轮廓检测，获取最外层矩形框的偏转角度
    show("img2", img2)
    show("img_contours2", img_contours2)
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # img = binary(img)
    img_cut = cut1(img.copy(), box,True)
    show("img_cut",img_cut)
    img_cut_rotate = rotate(img_cut, angle)
    show("img_cut_rotate",img_cut_rotate)

    img_cut_ = cv.resize(img_cut_rotate, (440, 140))
    img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7 = cut2(img_cut_)

    cv.imencode('.png', img_cut_rotate)[1].tofile(save_path + '/' + 'img_cut_rotate.png')
    cv.imencode('.png', img_cut_)[1].tofile(save_path + '/' + 'img_cut.png')
    cv.imencode('.png', img_cut1)[1].tofile(save_path + '/' + 'img_cut1.png')
    cv.imencode('.png', img_cut2)[1].tofile(save_path + '/' + 'img_cut2.png')
    cv.imencode('.png', img_cut3)[1].tofile(save_path + '/' + 'img_cut3.png')
    cv.imencode('.png', img_cut4)[1].tofile(save_path + '/' + 'img_cut4.png')
    cv.imencode('.png', img_cut5)[1].tofile(save_path + '/' + 'img_cut5.png')
    cv.imencode('.png', img_cut6)[1].tofile(save_path + '/' + 'img_cut6.png')
    cv.imencode('.png', img_cut7)[1].tofile(save_path + '/' + 'img_cut7.png')
    show("img_cut1", img_cut1)
    show("img_cut2", img_cut2)
    show("img_cut3", img_cut3)
    show("img_cut4", img_cut4)
    show("img_cut5", img_cut5)
    show("img_cut6", img_cut6)
    show("img_cut7", img_cut7)

    cv.waitKey(0)
    cv.destroyAllWindows()

def detction_and_cut(img):
    img = cv.resize(img, (512, 512))
    img_separate = separate_color_blue(img.copy())  # 提取蓝色框先
    try:
        img_contours, img2, box, angle = contour(img_separate, img.copy())  # 轮廓检测，获取最外层矩形框的偏转角度
    except ValueError:
        print("未检测到车牌！")
        return

    img_cut = cut1(img.copy(), box,True)
    img_cut_rotate = rotate(img_cut, angle)
    img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7 = cut2(img_cut_rotate)
    return [img_cut_rotate, img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7]


# 生成RGB色域，共16张图，每张图片大小为1024*1024
def RGB():
    save_RGB_path = "./RGB3"
    if os.path.exists(save_RGB_path)==False:
        os.makedirs(save_RGB_path)
    b,g,r = 0,0,0
    for i in range(16):
        img=[]
        for j in range(1024):
            w=[]
            for k in range(1024):
                h=[b,g,r]
                if g < 255:
                    g += 1
                elif r < 255:
                    r += 1
                    g = 0
                elif b < 255:
                    b += 1
                    r = 0
                    g = 0
                w.append(h)
            img.append(w)
        img = np.asarray(img)
        cv.imencode('.jpg', img)[1].tofile(save_RGB_path + '/' + 'RGB_'+ str(i+1) +'.jpg')




if __name__ == "__main__":
    cut_test_save()
    # RGB()
    # save_RGB_path = "./RGB"
    # for file in os.listdir(save_RGB_path):
    #     filed = save_RGB_path + '/' + file
    #     img = cv.imdecode(np.fromfile(filed, dtype=np.uint8), flags=cv.IMREAD_COLOR)
    #     img2 = separate_color_blue(img)
    #     cv.imencode('.jpg', img2)[1].tofile(filed + '_blue' +'.jpg')



