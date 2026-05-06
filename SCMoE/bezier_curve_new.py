import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

# 定义贝塞尔曲线函数
def bezier_curve(t, p0, p1, p2, p3):
    """ 计算三次贝塞尔曲线上的点 """
    return (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3

# 定义灰度变换函数，使用贝塞尔曲线
def apply_bezier_gray_transform(image):
    # 将图像转换为灰度图
    # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = image
    # 生成随机控制点 p1 和 p2，范围在 [0, 1] 之间
    p1 = random.uniform(-1, 1)
    p2 = random.uniform(-1, 1)

    # 设置固定的控制点 p0 和 p3
    p0 = -1  # 第一个控制点 [-1, -1]，这里用灰度值范围内的 -1
    p3 = 0   # 最后一个控制点 [0, 0]
    # 创建一个新的空白图像
    transformed_image = np.zeros_like(gray_image)

    # 遍历每个像素点，应用贝塞尔曲线
    for i in range(256):
        t = i / 255  # 灰度值归一化到 [0, 1]
        transformed_value = bezier_curve(t, p0, p1, p2, p3)  # 贝塞尔曲线变换
        transformed_image[gray_image == i] = int(transformed_value * 255)  # 映射变换

    return transformed_image

def main():
    # 读取图像
    image = cv2.imread('18.png')
    
    # 应用灰度变换
    transformed_image = apply_bezier_gray_transform(image)
    # 保存变换后的图像
    output_path = 'transformed_18.png'
    cv2.imwrite(output_path, transformed_image)
    print(f"Transformed image saved to {output_path}")

    # 显示原图和变换后的图像
    plt.figure(figsize=(10, 5))

    # 原图显示
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')

    # 变换后的图像显示
    plt.subplot(1, 2, 2)
    plt.imshow(transformed_image, cmap='gray')
    plt.title("Transformed Image (Bezier Curve)")
    plt.axis('off')

    # 展示图像
    plt.show()

# 确保只有在脚本直接运行时才执行 main()
if __name__ == "__main__":
    main()
