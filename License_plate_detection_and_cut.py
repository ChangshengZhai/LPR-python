import cv2 as cv
import numpy as np
import os



def show(name, img):
    # 显示图片
    cv.namedWindow(str(name), cv.WINDOW_AUTOSIZE)
    cv.imshow(str(name), img)


def binary(img):
    # 二值化处理去燥
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] < 130:
                img[i][j] = 0
            else:
                img[i][j] = 255
    return img


# 裁剪车牌
def cut1(img, box, flag):
    # 从轮廓出裁剪图片
    x, y = [], []
    for i in range(len(box)):
        x.append(box[i][0])
        y.append(box[i][1])
    x1, y1 = min(x), min(y)  # 获取左上角坐标
    x2, y2 = max(x), max(y)  # 获取右下角坐标
    x1, y1 = max([0, x1]), max([0, y1])
    x2, y2 = max([0, x2]), max([0, y2])
    # p为校验值
    p = 0
    if flag == False:
        p = int(len(img) * 0.05)
    img_cut = img[y1:y2, x1 + p:x2 - 2 * p, :]  # 切片裁剪图像
    return img_cut


# 裁剪出字符
def cut2(img_cut):
    img_cut = cv.resize(img_cut, (440, 140))
    img_cut = cv.cvtColor(img_cut, cv.COLOR_BGR2GRAY)
    img1 = img_cut[15:125, 15:61]
    img2 = img_cut[15:125, 72:118]
    img3 = img_cut[15:125, 151:197]
    img4 = img_cut[15:125, 208:254]
    img5 = img_cut[15:125, 265:311]
    img6 = img_cut[15:125, 322:368]
    img7 = img_cut[15:125, 379:425]
    return img1, img2, img3, img4, img5, img6, img7


def rect_cut(img, center, size, angle):
    center, size = tuple(map(int, center)), tuple(map(int, size))
    if size[0] < size[1]:
        angle -= 270
        w, h = size[1], size[0]
    else:
        w, h = size[0], size[1]
    height, width = img.shape[0], img.shape[1]
    M = cv.getRotationMatrix2D(center, angle, 1)
    img_rot = cv.warpAffine(img, M, (width, height))
    img_crop = cv.getRectSubPix(img_rot, tuple(map(int, [w, h])), center)
    return img_crop


def separate_color_blue(img):  # HSV阈值难以确定，暂时不用
    # 颜色提取
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)  # 色彩空间转换为hsv，便于分离
    # lower_hsv = np.array([106, 70, 20])  # 提取颜色的低值
    # high_hsv = np.array([180, 255, 255])  # 提取颜色的高值
    # lower_hsv = np.array([100, 90, 20])  # 提取颜色的低值    # 效果best
    # high_hsv = np.array([170, 255, 255])  # 提取颜色的高值
    lower_hsv = np.array([100, 60, 20])  # 提取颜色的低值
    high_hsv = np.array([170, 255, 255])  # 提取颜色的高值
    mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=high_hsv)
    # mask = binary(mask)
    # print("颜色提取完成")
    return mask


"""
vertices: 依次是矩阵的上下左右坐标
"""


def calculateWhiteRatio(img_rectangle):
    whiteRatioRet = 0
    whitePonitCnt = 0
    SumPonitCnt = 0
    img_t = img_rectangle
    # show("t", img_t)
    # cv.waitKey(0)
    for i in range(0, len(img_t)):
        for j in range(0, len(img_t[0])):
            SumPonitCnt += 1
            if img_t[i][j] == 255:
                whitePonitCnt += 1

    whiteRatioRet = whitePonitCnt / SumPonitCnt
    return whiteRatioRet


# 二值化图片中，白色填充大于80%，像素块大小合适，宽高比例在1.5--5之间的被认为有可能是车牌
def maybeLicencePlate(img_binary, rectangle):
    # 计算该矩阵连通域中，白色占比
    if calculateWhiteRatio(rect_cut(img_binary, rectangle[0], rectangle[1], rectangle[2])) < 0.35:
        return False

    return True


def contour(img1, img2):
    # 检测轮廓
    # img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    # show("kkk",img1)
    # cv.waitKey(0)
    ret, img1 = cv.threshold(img1, 127, 255, cv.THRESH_BINARY)

    contours, hier = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:  # 遍历轮廓
        rect = cv.minAreaRect(c)  # 生成最小外接矩形
        h = min([int(rect[1][0]), int(rect[1][1])])
        w = max([int(rect[1][0]), int(rect[1][1])])
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        if h > 10 and w > 20 and abs(rect[2]) < 30:  # 一定大小的目标矩阵才描边
            cv.drawContours(img1, [box], 0, (255, 255, 255), 8)

    # 补充完矩形框后，再检测一次，寻找车牌目标矩形
    contours, hier = cv.findContours(img1, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    count = 0
    max_w = 0
    max_h = 0
    max_angle = 0
    max_box, max_size, max_center = 0, 0, 0
    for c in contours:  # 遍历轮廓
        approx = cv.approxPolyDP(c, epsilon=5, closed=True)  # 多边拟合函数
        rect = cv.minAreaRect(approx)  # 生成最小外接矩形
        h = min([int(rect[1][0]), int(rect[1][1])])
        w = max([int(rect[1][0]), int(rect[1][1])])
        box = cv.boxPoints(rect)  # 计算最小面积矩形的坐标
        box = np.int0(box)  # 将坐标规范化为整数
        # 只保留需要的轮廓
        if h > len(img1[0]) * 0.35 and w > len(img1) * 0.4:
            continue
        if h < 20 or w < 10:
            continue
        if w / h > 4 or w / h <= 1.5:   # 这里可严格限制
            # cv.drawContours(img2, [box], 0, (0, 0, 255), 2)
            # show("img2", img2)
            # show("img1", img1)
            # cv.waitKey(0)
            continue

        if not maybeLicencePlate(img1, rect):
            continue
        # angle = rect[2]
        # if angle < 0 and angle >= -90:
        #     angle += 90
        # elif angle < -90 and angle >= -180:
        #     angle += 180
        # elif angle >
        # 筛选出最大的矩形对应的坐标, 矩形面积占比太大也不能选，默认车牌在整个图片中占比不超过30%，默认更加水平的矩形更容易是车牌
        if w > max_w and h > max_h:
            max_w, max_h = w, h
            max_box = box.copy()
            max_angle = rect[2]
            max_size = rect[1]
            max_center = rect[0]
            count += 1
        # print("angle", angle)
        # print("坐标", box)

    # show("img_1",img1)
    # cv.drawContours(img2, contours, -1, color=(0,0,255),thickness=2)
    # show("img1_contours",img2)
    # print("轮廓数量", count)
    # cv.waitKey()
    # print("max_box",max_box)
    # cv.waitKey()
    # show("max_box",img2)
    if count == 0:
        print("Error, can not find License Plate!")
        return 0, 0, 0, 0, 0, 0, 0
    cv.drawContours(img2, [max_box], 0, (0, 255, 0), 5)
    return 1, img1, img2, max_box, max_angle, max_size, max_center


def rotate(img, angle):
    # 旋转图片
    (h, w) = img.shape[:2]  # 获得图片高，宽
    center = (w // 2, h // 2)  # 获得图片中心点
    img_ratete = cv.getRotationMatrix2D(center, angle, 1)
    rotatedImg = cv.warpAffine(img, img_ratete, (w, h))
    return rotatedImg


def cut_test_save(img_path, save_path):
    bool = os.path.exists(save_path)
    if bool == False:
        os.makedirs(save_path)
    # 解决imread不能读取中文路径
    img = cv.imdecode(np.fromfile(img_path, dtype=np.uint8), flags=cv.IMREAD_COLOR)
    img = cv.resize(img, (512, int(512 * (len(img)/len(img[0])))))
    img_separate = separate_color_blue(img.copy())  # 提取蓝色框先
    cv.imencode('.png', img_separate)[1].tofile(save_path + '/' + 'test.jpg')
    show("img_separate", img_separate)

    debugFlag, img_contours2, img2, box, angle, size, center = contour(img_separate, img.copy())  # 轮廓检测，获取最外层矩形框的偏转角度
    if debugFlag == 0:
        return
    show("img2", img2)
    show("img_contours2", img_contours2)
    # img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # img = binary(img)

    # img_cut = cut1(img.copy(), box, True)
    # show("img_cut", img_cut)
    # img_cut_rotate = rotate(img_cut, angle)
    print(angle)
    img_cut_rotate = rect_cut(img.copy(), center, size, angle)
    show("img_cut_rotate", img_cut_rotate)

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
    img = cv.resize(img, (512, int(512 * (len(img) / len(img[0])))))
    img_separate = separate_color_blue(img.copy())  # 提取蓝色框先
    # try:
    #     img_contours, img2, box, angle, size, center = contour(img_separate, img.copy())  # 轮廓检测，获取最外层矩形框的偏转角度
    # except ValueError:
    #     print("未检测到车牌！")
    #     return
    # print("begin cutting!")
    # img_cut = cut1(img.copy(), box, True)
    # img_cut_rotate = rotate(img_cut, angle)
    flag, img_contours, img2, box, angle, size, center = contour(img_separate, img.copy())  # 轮廓检测，获取最外层矩形框的偏转角度
    img_cut_rotate = rect_cut(img.copy(), center, size, angle)

    img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7 = cut2(img_cut_rotate)
    return [img_cut_rotate, img_cut1, img_cut2, img_cut3, img_cut4, img_cut5, img_cut6, img_cut7]


# 生成RGB色域，共16张图，每张图片大小为1024*1024
def RGB():
    save_RGB_path = "./RGB3"
    if os.path.exists(save_RGB_path) == False:
        os.makedirs(save_RGB_path)
    b, g, r = 0, 0, 0
    for i in range(16):
        img = []
        for j in range(1024):
            w = []
            for k in range(1024):
                h = [b, g, r]
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
        cv.imencode('.jpg', img)[1].tofile(save_RGB_path + '/' + 'RGB_' + str(i + 1) + '.jpg')


if __name__ == "__main__":
    img_path = "./test_picture/"
    save_path = "./test_save"
    # img_path = './test_picture'
    for file in os.listdir(img_path):
        # try:
        #     image = cv.imdecode(np.fromfile(img_path + '/' + file, dtype=np.uint8), flags=cv.IMREAD_COLOR)
        # except ValueError:
        #     print("图片解析失败！")
        #     exit()
        # img = cv.resize(image, (512, 512))
        # show("image", img)
        # img_blue = separate_color_blue(img)
        # show("img_blue", img_blue)

        cut_test_save(img_path + '/' + file, save_path)



