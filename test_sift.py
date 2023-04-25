from functools import cmp_to_key
import cv2 as cv
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import time

################################
#############ARGUMENTS##########
################################
# 升采样参数
PRE_BLUR_SCALE = 2
PRE_BLUR_KSIZE = (0, 0)
# 高斯模糊/掩模参数
PRE_BLUR = 0.5
# 第0层的初始尺度，即第0层高斯模糊所使用的参数
SIFT_SIGMA = 1.6
# 高斯金字塔每组内的层数，S=3
NUM_DIFF_OCTAVES = 3
# 设置参数S，每组图像的层数，实际上这里的INTERVALS+3才是实际的S
NUM_INTVLS = 3
# 边界的像素宽度，检测过程中将忽略边界线中的极值点，即只检测边界线以内是否存在极值点
IMAGE_BORDER_WIDTH = 5
# 对比度阈值 |D(x)|，针对归一化后的图像，用来去除不稳定特征，|D(x)|<0.03此处使用RobHess此处的阈值使用的是0.04/S
CONTRAST_THRESHOLD=0.04
# 主曲率比值的阈值，用来去除边缘特征
RATIO_THRESHOLD=10
# 设置最大迭代次数
NUM_ATTMEPT_UNTIL_CONVERFENCE = 5
# 特征点方向赋值过程中，搜索邻域的半径为：3 * 1.5 * σ，分配方向时高斯权重参数因子，高斯加权（1.5σ）的半径
SCALE_FCTR = 1.5
# 计算特征描述子过程中，特征点周围的d*d个区域中，每个区域的宽度为m*σ个像素
RADIUS_FCTR = 3
# 特征点方向赋值过程中，梯度方向直方图中柱子(bin)的个数，0度~360度分36个柱，每10度一个柱
ORI_HIST_BINS = 36
# 特征点方向赋值过程中，梯度幅值达到最大值的80%则分裂为两个特征点，相当于主峰值80%能量的峰值
ORI_PEAK_RATIO = 0.8

#######cv.KeyPoint(*keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
class MyKeyPoint():
    def __init__(self, pt = (0,0), size = 0, orientation = 0, response = 0, octave = 0, class_id = None, des = None):
        self.pt = pt
        self.size = size
        self.orientation = orientation
        self.response = response
        self.octave = octave
        self.class_id = class_id
        self.des = des

def Upsampling(img):
    """对图像进行升采样，校准从取景器得到的图像，输入升采样后的图像
    :param img: input image
    :return: base image
    """
    img = img.astype(np.float32)
    img = cv.resize(img, PRE_BLUR_KSIZE, fx=PRE_BLUR_SCALE, fy=PRE_BLUR_SCALE, interpolation=cv.INTER_LINEAR)
    sigma_diff = np.sqrt(max((SIFT_SIGMA ** 2) - ((2 * PRE_BLUR) ** 2), 0.01))
    upsampling_img = cv.GaussianBlur(img, PRE_BLUR_KSIZE, sigmaX=sigma_diff, sigmaY=sigma_diff)
    return upsampling_img
def Get_Octaves_Num(base_img):
    """获取octaves的数量
    :param base_img: input image
    :return: octaves number
    """
    return int(np.round(np.log2(min(base_img.shape[:2]))) - NUM_DIFF_OCTAVES)
def GenerateGaussianKernels():
    """
    生成每一层的高斯核列表。
    """
    S = NUM_INTVLS + 3 # 一组6层图像
    k = 2 ** (1. / NUM_INTVLS)
    gaussian_kernels = np.zeros(S)
    gaussian_kernels[0] = SIFT_SIGMA
    for image_index in range(1, S):
        sigma_previous = (k ** (image_index - 1)) * SIFT_SIGMA
        sigma_total = k * sigma_previous
        gaussian_kernels[image_index] = np.sqrt(sigma_total ** 2 - sigma_previous ** 2)
    return gaussian_kernels
def GenerateGaussianImages(img, num_octaves, gaussian_kernels):
    """生成高斯金字塔
    :param img: input base_image
    :param num_octaves: octaves number
    :param gaussian_kernels: gaussian kernels
    """
    gaussian_imgs = []
    for i in range(num_octaves):
        gaussian_img_per_octave = []
        gaussian_img_per_octave.append(img)
        for kernel in gaussian_kernels[1:]:
            img = cv.GaussianBlur(img, (0, 0), sigmaX=kernel, sigmaY=kernel)
            gaussian_img_per_octave.append(img)
        gaussian_imgs.append(gaussian_img_per_octave)
        # 该组的第三层图像作为下一组的基础图像
        octave_base_img = gaussian_img_per_octave[-3]
        # 降采样
        img = cv.resize(octave_base_img, (int(octave_base_img.shape[1] / 2), int(octave_base_img.shape[0] / 2)), interpolation=cv.INTER_NEAREST)
    return np.array(gaussian_imgs, dtype=object)
def GenerateDoGImages(gaussian_img):
    """Generate Difference-of-Gaussians image pyramid
    """
    DoG_images = []
    for img in gaussian_img:
        dog_images_in_octave = []
        for st, nd in zip(img, img[1:]):
            # 这里使用cv.subtract()函数，而不是nd - st
            dog_images_in_octave.append(cv.subtract(nd, st))
        DoG_images.append(dog_images_in_octave)
    return np.array(DoG_images, dtype=object)
def FindAllKeyPoints(gaussian_imgs, DoG_imgs):
    """
    在图像金字塔中找到所有尺度空间极值点的像素位置
    """
    threshold = np.floor(0.5 * CONTRAST_THRESHOLD / NUM_INTVLS * 255)  # from OpenCV implementation
    keypoints = []
    cnt_local_extrema = 0
    cnt_accruate_keypoints = 0
    for octave_index, DoG_imgs_in_octave in enumerate(DoG_imgs):
        # 将DOG图像中的三张图像进行zip，然后取中间那张图像的中心点进行比较
        three_zip_img = zip(DoG_imgs_in_octave, DoG_imgs_in_octave[1:], DoG_imgs_in_octave[2:])
        for layer_index, img_truple in enumerate(three_zip_img):
            # 遍历图像进行比较
            for i in range(IMAGE_BORDER_WIDTH, img_truple[0].shape[0] - IMAGE_BORDER_WIDTH):
                for j in range(IMAGE_BORDER_WIDTH, img_truple[0].shape[1] - IMAGE_BORDER_WIDTH):
                    # 判断是否是局部极值点
                    if IsLocalExtrema(*(Find3x3x3Cube(*(img_truple), i, j)), threshold):
                        cnt_local_extrema += 1
                        local_extrema = IsAccurateKeyPoints(i, j, layer_index + 1, octave_index, DoG_imgs_in_octave)
                        if local_extrema is not None:
                            cnt_accruate_keypoints += 1
                            # 方向分配
                            keypoint, local_layer_index = local_extrema
                            cal_img = gaussian_imgs[octave_index][local_layer_index]
                            keypoints_with_orientations = AssignOrientationToKeypoints(keypoint, octave_index,cal_img)
                            for kk in keypoints_with_orientations:
                                keypoints.append(kk)
    print(f"cnt_local_extrema: {cnt_local_extrema}")
    print(f"cnt_local_extrema: {cnt_accruate_keypoints}")
    return keypoints
def FindAllKeyPointsMuti(gaussian_imgs, DoG_imgs):
    """
    在图像金字塔中找到所有尺度空间极值点的像素位置
    """
    keypoints_list = []
    threshold = np.floor(0.5 * CONTRAST_THRESHOLD / NUM_INTVLS * 255)  # from OpenCV implementation
    cnt_local_extrema = 0
    cnt_accruate_keypoints = 0
    args = []
    for octave_index, DoG_imgs_in_octave in enumerate(DoG_imgs):
        # 将DOG图像中的三张图像进行zip，然后取中间那张图像的中心点进行比较
        three_zip_img = zip(DoG_imgs_in_octave, DoG_imgs_in_octave[1:], DoG_imgs_in_octave[2:])
        for layer_index, img_truple in enumerate(three_zip_img):
            # 遍历图像进行比较
            for i in range(IMAGE_BORDER_WIDTH, img_truple[0].shape[0] - IMAGE_BORDER_WIDTH):
                for j in range(IMAGE_BORDER_WIDTH, img_truple[0].shape[1] - IMAGE_BORDER_WIDTH):
                    args.append((i,j,img_truple,threshold,octave_index,layer_index,gaussian_imgs,DoG_imgs_in_octave))
    print(len(args))
    pool = Pool(processes=8)
    results = pool.starmap(MutiTask, args)
    pool.close()
    pool.join()
    for result in results:
        if result is not None:
            for kk in result:
                keypoints_list.append(kk)
    print(f"cnt_local_extrema: {cnt_local_extrema}")
    print(f"cnt_local_extrema: {cnt_accruate_keypoints}")
    return keypoints_list
def MutiTask(i,j,img_truple,threshold,octave_index,layer_index,gaussian_imgs,DoG_imgs_in_octave):
    """
    """
    if IsLocalExtrema(*(Find3x3x3Cube(*(img_truple), i, j)), threshold):
        # cnt_local_extrema += 1
        local_extrema = IsAccurateKeyPoints(i, j, layer_index + 1, octave_index, DoG_imgs_in_octave)
        if local_extrema is not None:
            # cnt_accruate_keypoints += 1
            # 方向分配
            keypoint, local_layer_index = local_extrema
            cal_img = gaussian_imgs[octave_index][local_layer_index]
            keypoints_with_orientations = AssignOrientationToKeypoints(keypoint, octave_index,cal_img)
            return keypoints_with_orientations
    return None
def Find3x3x3Cube(st, nd, rd, i, j):
    """
    从3xoctxlayer的zip图像中取出中心点的3x3x3的img Grid
    """
    return st[i-1:i+2,j-1:j+2],\
           nd[i-1:i+2,j-1:j+2],\
           rd[i-1:i+2,j-1:j+2]
def IsLocalExtrema(st, nd, rd, threshold):
    """
    根据3x3x3的数组判断中心点是否是极值点，如果是极值点则返回True，否则返回False，并且极值点的阈值为threshold
    """
    c = nd[1, 1]
    # 把这个3x3x3的矩阵展开成一个一维数组，然后找出最大值和最小值
    if abs(c) > threshold:
        if c > 0:
            return np.all(c >= st) and np.all(c >= rd) and np.all(c >= nd[0, :]) and np.all(c >= nd[2, :]) and c >= nd[1, 0] and c >= nd[1, 2]
        elif c < 0:
            return np.all(c <= st) and np.all(c <= rd) and np.all(c <= nd[0, :]) and np.all(c <= nd[2, :]) and c <= nd[1, 0] and c <= nd[1, 2]
    return False
def IsOutOfBorder(img, i, j, layer_index):
    """
    判断像素点是否在图像边界之外
    """
    if i < IMAGE_BORDER_WIDTH or i >= img[0] - IMAGE_BORDER_WIDTH or \
       j < IMAGE_BORDER_WIDTH or j >= img[1] - IMAGE_BORDER_WIDTH or \
       layer_index < 1 or layer_index > NUM_INTVLS:
        return True
    return False
def IsAccurateKeyPoints(i, j, layer_index, octave_index, DoG_imgs_in_octave):
    """
    通过对每个极值点的邻域进行二次拟合来迭代地改进尺度空间极值点的像素位置
    """
    is_out_imgs = False
    for cnt_attempt in range(NUM_ATTMEPT_UNTIL_CONVERFENCE): # 5次迭代
        # 将像素值缩放到[0,1]来应用Lowe的阈值
        img_cube = Find3x3x3Cube(*(DoG_imgs_in_octave[layer_index - 1:layer_index + 2]), i, j)
        pixel_cube = np.stack([*(img_cube)]).astype('float32') / 255.
        # 计算该cube的dD/dx,dD/dy,dD/d(sigma),返回为列向量dD/dX
        gradient = GetFirstDifference(pixel_cube)
        # 用离散值近似计算出三维hessian矩阵,即公式中d2D/dX2
        hessian = GetSecondDifference(pixel_cube)
        # 解方程组,返回的回归系数、残差平方和、自变量X的秩、X的奇异值，我们只需要回归系数即可
        X_hat = -np.linalg.lstsq(hessian, gradient, rcond=None)[0]
        # 如果回归系数的绝对值都小于0.5，则认为该点已经收敛，否则继续迭代
        if abs(X_hat[0]) < 0.5 and abs(X_hat[1]) < 0.5 and abs(X_hat[2]) < 0.5:
            break
        j += int(round(X_hat[0]))
        i += int(round(X_hat[1]))
        layer_index += int(round(X_hat[2]))
        # 确保插值后的pixel_cube完全位于图像内
        is_out_imgs = IsOutOfBorder(DoG_imgs_in_octave[0].shape, i, j, layer_index)
        if is_out_imgs:
            break
    # 超出边界，或者迭代次数超过5次，或者像素点超出图像边界
    if is_out_imgs or cnt_attempt >= NUM_ATTMEPT_UNTIL_CONVERFENCE - 1:
        return None
    # 最后得到更新后的极值点,带入一阶泰勒展开
    pix_x_hat = pixel_cube[1, 1, 1] + 0.5 * np.dot(gradient, X_hat)
    # 消除边缘效应
    # 首先要保证该点仍然大于阈值
    if abs(pix_x_hat) * NUM_INTVLS >= CONTRAST_THRESHOLD:
        # 计算曲率，首先获得hessian矩阵的前两行两列，计算矩阵的迹和行列式
        xy_hessian = hessian[:2, :2]
        trace = np.trace(xy_hessian)
        det = np.linalg.det(xy_hessian)
        # 判断曲率是否小于阈值，如果小于阈值，则认为该点是边缘点
        # 计算公式为：(r+1)^2 / r < (tr(H))^2 / det(H) ---> (r+1)^2 * det(H) < (tr(H))^2 * r 使用乘法会更快一些
        if det > 0 and RATIO_THRESHOLD * (trace ** 2) < ((RATIO_THRESHOLD + 1) ** 2) * det:
            # 生成关键点
            keypoint = MyKeyPoint()
            # 关键点坐标为：((j + X_hat[0]) * (2^octave_index), (i + X_hat[1]) * (2^octave_index))
            keypoint.pt = ((j + X_hat[0]) * (2 ** octave_index), (i + X_hat[1]) * (2 ** octave_index))
            # 由于opencv中octave的存储方式是把octave_index和layer_index放在一个int32中，所以需要进行一些处理
            keypoint.octave = octave_index + layer_index * (2 ** 8) + int(round((X_hat[2] + 0.5) * 255)) * (2 ** 16)
            # octave_index + 1是因为输入图像被放大了一倍
            keypoint.size = SIFT_SIGMA * (2 ** ((layer_index + X_hat[2]) / np.float32(NUM_INTVLS))) * (2 ** (octave_index + 1))
            # 存储像素到response中
            keypoint.response = abs(pix_x_hat)
            return keypoint, layer_index
    return None
def GetFirstDifference(pix):
    """使用中心差分公式近似计算3x3x3数组中心像素[1,1,1]的梯度，h = 1
    """
    # dx = 1/2 * (D(x+1, y, s) -  D(x-1, y, s))
    # dy = 1/2 * (D(x, y+1, s) -  D(x, y-1, s))
    # ds = 1/2 * (D(x, y, s+1) -  D(x, y, s-1))
    dx = 0.5 * (pix[1, 1, 2] - pix[1, 1, 0])
    dy = 0.5 * (pix[1, 2, 1] - pix[1, 0, 1])
    ds = 0.5 * (pix[2, 1, 1] - pix[0, 1, 1])
    return np.array([dx, dy, ds])
def GetSecondDifference(pix):
    """使用heissen矩阵近似计算3x3x3数组中心像素[1,1,1]的梯度，h = 1
    """
    # dxx = D(x+1, y, s) - 2 * D(x, y, s) + D(x-1, y, s)
    # dyy = D(x, y+1, s) - 2 * D(x, y, s) + D(x, y-1, s)
    # dss = D(x, y, s+1) - 2 * D(x, y, s) + D(x, y, s-1)
    # dxy = 1/4 * (D(x+1, y+1, s) - D(x+1, y-1, s) - D(x-1, y+1, s) + D(x-1, y-1, s))
    # dxs = 1/4 * (D(x+1, y, s+1) - D(x+1, y, s-1) - D(x-1, y, s+1) + D(x-1, y, s-1))
    # dys = 1/4 * (D(x, y+1, s+1) - D(x, y+1, s-1) - D(x, y-1, s+1) + D(x, y-1, s-1))
    dxx = pix[1, 1, 2] - 2 * pix[1, 1, 1] + pix[1, 1, 0]
    dyy = pix[1, 2, 1] - 2 * pix[1, 1, 1] + pix[1, 0, 1]
    dss = pix[2, 1, 1] - 2 * pix[1, 1, 1] + pix[0, 1, 1]
    dxy = 0.25 * (pix[1, 2, 2] - pix[1, 2, 0] - pix[1, 0, 2] + pix[1, 0, 0])
    dxs = 0.25 * (pix[2, 1, 2] - pix[2, 1, 0] - pix[0, 1, 2] + pix[0, 1, 0])
    dys = 0.25 * (pix[2, 2, 1] - pix[2, 0, 1] - pix[0, 2, 1] + pix[0, 0, 1])
    return np.array([[dxx, dxy, dxs],
                  [dxy, dyy, dys],
                  [dxs, dys, dss]])
def CalRawHistogram(keypoint,radius,gauss_img,weight_factor,octave_index):
    """计算关键点的直方图
    """
    raw_his = np.zeros(ORI_HIST_BINS)
    img_shape = gauss_img.shape
    for i in range(-radius, radius + 1):
        region_y = int(round(keypoint.pt[1] / np.float32(2 ** octave_index))) + i
        if region_y > 0 and region_y < img_shape[0] - 1:
            for j in range(-radius, radius + 1):
                region_x = int(round(keypoint.pt[0] / np.float32(2 ** octave_index))) + j
                if region_x > 0 and region_x < img_shape[1] - 1:
                    # dx = D(x+1, y) - D(x-1, y)
                    dx = gauss_img[region_y, region_x + 1] - gauss_img[region_y, region_x - 1]
                    # dy = D(x, y+1) - D(x, y-1)
                    dy = gauss_img[region_y - 1, region_x] - gauss_img[region_y + 1, region_x]
                    # 梯度的模值
                    gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                    # 梯度的方向
                    gradient_orientation = np.rad2deg(np.arctan2(dy, dx))
                    # 乘以一个权重，高斯函数的常数项可以省略
                    weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                    histogram_index = int(round(gradient_orientation * ORI_HIST_BINS / 360.))
                    # 梯度值乘以高斯权重
                    raw_his[histogram_index % ORI_HIST_BINS] += weight * gradient_magnitude
    return raw_his
def SmoothHistogram(raw_his):
    """平滑直方图
    """
    smooth_his = np.zeros(ORI_HIST_BINS)
    for n in range(ORI_HIST_BINS):
        smooth_his[n] = (6 * raw_his[n] +
                         4 * (raw_his[n - 1] + raw_his[(n + 1) % ORI_HIST_BINS]) +
                         raw_his[n - 2] +
                         raw_his[(n + 2) % ORI_HIST_BINS]
                         ) / 16.
    return smooth_his
def CalOrientationForPeaks(ori_max, ori_peaks, smooth_his, keypoint):
    ori_keypoints = []
    for i in ori_peaks:
        peak_value = smooth_his[i]
        # 设置峰值为80%的方向直方图最大值
        if peak_value >= ORI_PEAK_RATIO * ori_max:
            l_v = smooth_his[(i - 1) % ORI_HIST_BINS]
            r_v = smooth_his[(i + 1) % ORI_HIST_BINS]
            # idx = i + 0.5 * (l_v - r_v) / (l_v - 2 * peak_value + r_v)
            interpolated_peak_index = (i + 0.5 * (l_v - r_v) / (l_v - 2 * peak_value + r_v)) % ORI_HIST_BINS
            # 更新关键点的方向
            orientation = 360. - interpolated_peak_index * 360. / ORI_HIST_BINS
            if abs(orientation - 360.) < 1e-7:
                orientation = 0
            new_keypoint = MyKeyPoint(keypoint.pt, keypoint.size, orientation, keypoint.response, keypoint.octave)
            ori_keypoints.append(new_keypoint)
    return ori_keypoints
def AssignOrientationToKeypoints(keypoint, octave_index, gauss_img):
    """计算关键点的方向
    """
    # 当前关键点所在的尺度 σ_oct = 1.5 * size / 2^(octave_index+1)
    oct = SCALE_FCTR * keypoint.size / np.float32(2 ** (octave_index + 1))
    # 3*1.5oct = 4.5oct
    radius = int(round(RADIUS_FCTR * oct))
    weight_factor = -0.5 / (oct ** 2)
    # 计算原始方向直方图
    raw_his = CalRawHistogram(keypoint,radius,gauss_img,weight_factor,octave_index)
    # 直方图的平滑
    smooth_his = SmoothHistogram(raw_his)
    # 取方向直方图中最大值作为该关键点的主方向
    orientation_max = max(smooth_his)
    # 峰值的定义为，比左右相邻的两个方向值都大，作为辅方向
    orientation_peaks = np.where(np.logical_and(smooth_his > np.roll(smooth_his, 1), smooth_his > np.roll(smooth_his, -1)))[0]
    # 为每个峰值分配方向
    assigned_ori_keypoints = CalOrientationForPeaks(orientation_max, orientation_peaks, smooth_his, keypoint)
    ##############################################Debug用########
    # ry = int(round(keypoint.pt[1] / np.float32(2 ** octave_index)))
    # rx = int(round(keypoint.pt[0] / np.float32(2 ** octave_index)))
    # # print(ry - radius, ry + radius, rx - radius, rx + radius)
    # # # 峰值的定义为，比左右相邻的两个方向值都大
    #
    # plt.imshow(gauss_img[ry - radius: ry + radius,rx - radius: rx + radius - 2])
    # for i in range(len(ori)):
    #     height, width = gauss_img[ry - radius: ry + radius,rx - radius: rx + radius - 2].shape
    #     # Calculate the endpoints of the line
    #     angle = ori[0] # Specify the angle in radians
    #     # plt.title(f"angle = {int(angle)}", fontsize=2)
    #     angle = np.deg2rad(angle)
    #     x0, y0 = width // 2, height // 2  # Specify the center point of the line
    #     length = min(height, width) // 2  # Specify the length of the line
    #     # x1 = x0 + length * np.cos(angle)
    #     # y1 = y0 + length * np.sin(angle)
    #     # x2 = x0 - length * np.cos(angle)
    #     # y2 = y0 - length * np.sin(angle)
    #     # plt.plot([x0, x2], [y0, y2], color='yellow', linewidth=2)
    #     plt.arrow(x0, y0, length*0.8 * -np.cos(angle), length * -np.sin(angle),
    #       head_width=1, head_length=1, fc='yellow', ec='yellow')
    #############################################################
    return assigned_ori_keypoints

def convertKeypointsToInputImageSize(keypoints):
    """
    由于在高斯金字塔中，每一层图像的尺寸都是上一层的一半，因此需要将关键点的坐标、尺度、所在的金字塔层数都乘以2
    """
    converted_keypoints = []

    for keypoint in keypoints:
        keypoints_new = cv.KeyPoint(
            * tuple(0.5 * np.array(keypoint.pt)),
            keypoint.size * 0.5,
            keypoint.angle,
            keypoint.response,
            ((keypoint.octave & ~255) | ((keypoint.octave - 1) & 255))
        )
        converted_keypoints.append(keypoints_new)
    return converted_keypoints


if __name__ == '__main__':
    raw_image = cv.imread('box_in_scene.png', 0)
    base_image = Upsampling(raw_image)
    print(f"base_image.shape: {base_image.shape},raw_image.shape: {raw_image.shape}")
    num_octaves = Get_Octaves_Num(base_image)
    print(f"num_octaves: {num_octaves}")
    guassian_kernels = GenerateGaussianKernels()
    print(f"guassian_kernels: {guassian_kernels}")
    gaussian_imgs = GenerateGaussianImages(base_image, num_octaves, guassian_kernels)
    DoG_imgs = GenerateDoGImages(gaussian_imgs)
    start = time.time()
    keypoints = FindAllKeyPointsMuti(gaussian_imgs, DoG_imgs)
    end = time.time()
    print(f"FindAllKeyPoints time = {end - start}")
    print(f"keypoints: {len(keypoints)}")
    cvKeyPoints = []
    for keypoint in keypoints:
        cvKeyPoints.append(cv.KeyPoint(keypoint.pt[0],keypoint.pt[1], keypoint.size, keypoint.orientation, keypoint.response, keypoint.octave))
    cvKeyPoints = convertKeypointsToInputImageSize(cvKeyPoints)
    img1 = cv.drawKeypoints(raw_image, cvKeyPoints, None, color=(0, 255, 0), flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # plt.imshow(img1)
    plt.imsave('box_in_scene_keypoints.png', img1)