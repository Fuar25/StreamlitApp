import cv2
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import io

def variance_of_laplacian(image):
    """计算拉普拉斯方差，用于评估图片清晰度"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def calculate_brightness(image):
    """计算图片的平均亮度"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def plot_histogram(image):
    """绘制并返回图片的直方图"""
    color = ('b','g','r')
    for i,col in enumerate(color):
        hist = cv2.calcHist([image],[i],None,[256],[0,256])
        plt.plot(hist,color = col)
        plt.xlim([0,256])
    plt.xlabel('Intensity')
    plt.ylabel('Count')
    plt.title('Grayscale Histogram')
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

def display_image_with_histogram(image):
    """展示图片及其直方图"""
    st.image(image, caption="上传的图片", use_column_width=True)

    # 将OpenCV格式的BGR图像转换为RGB以便Matplotlib绘制
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    histogram_buf = plot_histogram(img_rgb)
    st.image(histogram_buf, caption="图片直方图", use_column_width=True)

def run_image_analysis_page():
    st.title('图片分析工具')

    # 允许用户上传图片
    uploaded_file = st.file_uploader("选择一张图片", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # 读取上传的图片
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # 展示图片及其参数
        display_image_with_histogram(image)
        display_image_parameters(image)

def display_image_parameters(image):
    """展示图片的亮度参数和拉普拉斯方差"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = calculate_brightness(image)
    laplacian_variance = variance_of_laplacian(gray)

    st.write(f"亮度参数: {brightness:.2f}")
    st.write(f"拉普拉斯方差: {laplacian_variance:.2f}")

if __name__ == '__main__':
    run_image_analysis_page()
