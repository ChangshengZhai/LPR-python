import cv2 as cv
import os
import numpy as np

path = './chinese_license_picture_data2'
save_path1 = './chines_font_data'
save_path2 = './single_character_data'
Chinese_font = ['冀', '新', '鄂', '宁', '桂', '黑', '湘', '皖', '云', '豫', '蒙', '赣'
        , '吉', '辽', '苏', '甘', '晋', '浙', '闽', '渝', '贵', '陕', '粤', '川', '鲁', '琼'
        , '青', '藏', '京', '津', '沪']
# 24个英文字符除I和O
English_font = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R'
        ,'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
number = ['0','1','2','3','4','5','6','7','8','9']


# 一定范围内的随机旋转、高斯模糊、二值化、剪裁拉伸
def preprocessing():

        return 0

def main():
        bool = os.path.exists(save_path1)
        print(bool)
        if bool == False:
                os.makedirs(save_path1)
        bool = os.path.exists(save_path2)
        print(bool)
        if bool == False:
                os.makedirs(save_path2)

        for file in os.listdir(path):
                chinese_bool = False
                i = 0
                pathd = path + '/' + file
                for filed in os.listdir(pathd):
                        character1_bool = False
                        j = 0
                        pathdd = pathd + '/' + filed
                        for filedd in os.listdir(pathdd):
                                character2_bool = False
                                imgname = pathdd + '/' + filedd
                                # 解决imread不能读取中文路径
                                img = cv.imdecode(np.fromfile(imgname,dtype=np.uint8),flags=0)
                                # 裁剪并保存中文字符，注：裁剪范围事先已经算好
                                if chinese_bool==False:
                                        img_chinese = img[25:115,15:61]
                                        cv.imencode('.png', img_chinese)[1].tofile(save_path1 + '/' + '{}.png'.format(file))
                                # 裁剪并保存车牌序号字符
                                if i == 0 and character1_bool == False:
                                        img_character1 = img[25:115, 72:118]
                                        cv.imencode('.png', img_character1)[1].tofile(save_path2 + '/' + '{}.png'.format(filed))
                                if i == 0 and j == 0 and character2_bool == False:
                                        img_character2 = img[25:115, 151:197]
                                        cv.imencode('.png', img_character2)[1].tofile(save_path2 + '/' + '{}.png'.format(filedd[2]))

                                # 保存裁剪的图片
                                chinese_bool = True
                        j+=1
                i+=1



if __name__ == '__main__':
    main()