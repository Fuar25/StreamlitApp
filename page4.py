import cv2
import numpy as np
import streamlit as st
import os
from PIL import Image

def variance_of_laplacian(image):
    """计算拉普拉斯方差，用于评估图片清晰度"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_brightness(image):
    """计算图片的平均亮度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def color_histogram_complexity(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    entropy = shannon_entropy(hist)
    return entropy

def shannon_entropy(hist):
    hist = np.where(hist != 0, hist, 1e-12)  # 避免除零错误
    return -np.sum(hist * np.log2(hist))

def is_image_too_dark(image, threshold=70):
    """判断图片是否太暗"""
    avg = np.mean(image)
    return avg < threshold

def is_image_too_light(image, threshold=220):
    """判断图片是否太亮"""
    avg = np.mean(image)
    return avg > threshold

def compare_images(image1_path, image2_path, threshold=1000):
    """
    比较两幅图片的相似度
    :param image1_path: 图片1路径
    :param image2_path: 图片2路径
    :param threshold: 相似度阈值
    :return: True 如果图片相似，否则 False
    """
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    diff = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    mse = diff / (img1.shape[0] * img1.shape[1] * 3)

    return mse < threshold

def delete_similar_images(directory, threshold=1000):
    """
    删除目录中高度相似的图片
    :param directory: 图片所在目录
    :param threshold: 相似度阈值
    """
    images = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
    images_num = list(range(len(images)))

    for i in images_num:
        for j in range(i + 1, len(images)):
            image1_path = os.path.join(directory, images[i])
            image2_path = os.path.join(directory, images[j])

            if compare_images(image1_path, image2_path, threshold):
                print(f"Deleting {image2_path}")
                os.remove(image2_path)

        images = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png')]
        images_num = list(range(len(images)))

def process_images(folder_path):
    """处理文件夹中的所有图片，删除不符合条件的图片，并重命名符合条件的图片"""
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)
    valid_images = []
    valid_image_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, image_file in enumerate(images):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # 转为灰度图以简化清晰度检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检查图片是否太暗或太亮
        if (is_image_too_dark(gray) or is_image_too_light(gray)):
            os.remove(image_path)
            continue

        # 计算拉普拉斯方差
        fm = variance_of_laplacian(gray)

        # 设置清晰度阈值
        if fm < 37:  # 这个值可以根据实际情况调整
            os.remove(image_path)
            continue

        # 将符合条件的图片添加到列表中
        valid_images.append((image_file, image))
        valid_image_count += 1

        # 重命名图片
        new_image_name = f"{valid_image_count}.jpg"
        new_image_path = os.path.join(folder_path, new_image_name)
        os.rename(image_path, new_image_path)

        # 更新进度条
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
        status_text.text(f"正在处理第 {i + 1}/{total_images} 张图片...")

    return valid_images

def run_image_selection_page():
    st.title('图片筛选工具')

    # 获取文件夹路径
    folder_path = st.text_input("请输入文件夹路径：")

    # 添加一个按钮
    if st.button('开始处理'):
        if folder_path and os.path.isdir(folder_path):
            # 显示进度条
            st.write("开始处理图片...")
            valid_images = process_images(folder_path)

            # 删除相似图片
            delete_similar_images(folder_path, threshold=1200)

            # 展示符合条件的图片
            st.write("符合条件的图片：")
            for image_file, image in valid_images:
                # 计算亮度参数
                brightness = calculate_brightness(image)
                # 展示图片及其参数
                st.image(image, caption=f'{image_file} - 清晰度得分: {variance_of_laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):.2f}, 亮度参数: {brightness:.2f}', use_column_width=True)

if __name__ == '__main__':
    run_image_selection_page()
