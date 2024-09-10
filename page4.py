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

def is_image_too_dark(image, threshold=50):
    """判断图片是否太暗"""
    avg = np.mean(image)
    return avg < threshold

def is_image_too_light(image, threshold=220):
    """判断图片是否太亮"""
    avg = np.mean(image)
    return avg > threshold

def process_images(folder_path):
    """处理文件夹中的所有图片，删除不符合条件的图片"""
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)
    valid_images = []
    invalid_images = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, image_file in enumerate(images):
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        if image is None:
            continue

        # 转为灰度图以简化清晰度检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检查图片是否太暗
        if is_image_too_dark(gray):
            invalid_images.append(image_file)
            continue

        if is_image_too_light(gray):
            invalid_images.append(image_file)
            continue

        # 计算拉普拉斯方差
        fm = variance_of_laplacian(gray)

        # 设置清晰度阈值
        if fm < 40:  # 这个值可以根据实际情况调整
            invalid_images.append(image_file)
            continue

        # 将符合条件的图片添加到列表中
        valid_images.append((image_file, image))

        # 更新进度条
        progress = (i + 1) / total_images
        progress_bar.progress(progress)
        status_text.text(f"正在处理第 {i + 1}/{total_images} 张图片...")

    # 删除不符合条件的图片
    for image_file in invalid_images:
        os.remove(os.path.join(folder_path, image_file))

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

            # 展示符合条件的图片
            st.write("符合条件的图片：")
            for image_file, image in valid_images:
                # 计算亮度参数
                brightness = calculate_brightness(image)
                # 展示图片及其参数
                st.image(image, caption=f'{image_file} - 清晰度得分: {variance_of_laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)):.2f}, 亮度参数: {brightness:.2f}', use_column_width=True)

if __name__ == '__main__':
    run_image_selection_page()
