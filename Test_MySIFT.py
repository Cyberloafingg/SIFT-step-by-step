import MySIFT
import cv2 as cv
# 使用Opencv进行特征点提取
def detect_sift_opencv(img):
    sift = cv.SIFT_create()
    kp = sift.detect(img, None)
    kp, des = sift.compute(img, kp)
    print(f"opencv找到的该图片特征点个数：{len(kp)}")
    return kp, des
if __name__ == '__main__':
    # 提供一个函数SIFTProcess，输入图片路径，输出关键点和描述子以及raw_img
    # 注意是否开启多进程加速，请确保cpu核心数量大于2
    # 绘制图像请确保is_show_img为True，该图像由matplotlib绘制，请确保IDE可以正确显示
    img = "xiaogong.jpg"
    kp,des,img =  MySIFT.SIFTProcess(img,is_use_multiprocessing=True,is_show_img=True)
    detect_sift_opencv(img)
    # 提供一个函数JudgeGoodMatch，输入两张图片的路径，绘制匹配后的图像
    MySIFT.JudgeGoodMatch('xiaogonghead.jpg', 'xiaogong.jpg')