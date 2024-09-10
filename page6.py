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

def display_image_parameters(image):
    """展示图片的亮度参数和拉普拉斯方差"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(image)
    laplacian_variance = variance_of_laplacian(gray)

    st.image(image, caption="上传的图片", use_column_width=True)
    st.write(f"亮度参数: {brightness:.2f}")
    st.write(f"拉普拉斯方差: {laplacian_variance:.2f}")

def run_image_parameter_page():
    st.title('图片参数展示')

    # 允许用户上传图片
    uploaded_file = st.file_uploader("选择一张图片", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # 读取上传的图片
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # 展示图片及其参数
        display_image_parameters(image)

if __name__ == '__main__':
    run_image_parameter_page()
