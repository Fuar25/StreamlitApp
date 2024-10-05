import streamlit as st
import cv2
import os
import tempfile
import numpy as np
from PIL import Image

def play_video(video_file):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        st.error("Error opening video file")
        return

    # 在Streamlit中显示视频
    video_container = st.empty()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 显示视频帧
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_container.image(frame)

        if st.button('Stop Video', key=f'stop_video_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}'):
            break

    cap.release()
    video_container.empty()  # 清空容器

def extract_frames(video_file, output_dir):
    cap = cv2.VideoCapture(video_file)
    count = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(output_dir, f'frame_{count}.jpg'), frame)
        count += 1
    cap.release()

def load_image(image_path):
    """加载图片"""
    return cv2.imread(image_path)

def compare_images(image1_path, image2_path, threshold=1000):
    """
    比较两幅图片的相似度
    :param image1_path: 图片1路径
    :param image2_path: 图片2路径
    :param threshold: 相似度阈值
    :return: True 如果图片相似，否则 False
    """
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)

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
        print(i)
        print(len(images))

def variance_of_laplacian(image):
    """计算拉普拉斯方差，用于评估图片清晰度"""
    return cv2.Laplacian(image, cv2.CV_64F).var()

def is_image_too_dark(image, threshold=64):
    """判断图片是否太暗"""
    avg = np.mean(image)
    return avg < threshold

def is_image_too_light(image, threshold=220):
    """判断图片是否太亮"""
    avg = np.mean(image)
    return avg > threshold

def process_images(folder_path):
    """处理文件夹中的所有图片，删除不符合条件的图片，并重命名符合条件的图片"""
    images = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(images)
    valid_images = []
    valid_image_count = 0

    progress_bar = st.progress(0)
    status_text = st.empty()

def read_image(path):
    """读取图像并转换为灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {path}")
    return img

def detect_edges(img):
    """使用 Canny 边缘检测算法检测图像边缘"""
    edges = cv2.Canny(img, threshold1=50, threshold2=150)
    return edges

def calculate_edge_density(edges):
    """计算边缘密度"""
    total_pixels = edges.size
    edge_pixels = cv2.countNonZero(edges)
    edge_density = edge_pixels / total_pixels
    return edge_density

def is_low_complexity_image(edge_density, threshold=0.01):
    """判断图像复杂度是否低于阈值"""
    return edge_density < threshold

def filter_images(directory_path, threshold=0.01):
    """遍历目录中的所有图像，删除复杂度较低的图像"""
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory_path, filename)
            try:
                img = read_image(img_path)
                edges = detect_edges(img)
                edge_density = calculate_edge_density(edges)

                if is_low_complexity_image(edge_density, threshold):
                    os.remove(img_path)
                    print(f"Deleted low complexity image: {filename}, {edge_density}")
                else:
                    print(f"Not deleted low complexity image: {filename}, {edge_density}")
            except FileNotFoundError as e:
                print(e)

def main():
    temp_dir = tempfile.mkdtemp()
    uploaded_video = st.file_uploader('Upload a video file', type=['avi', 'mp4', 'mov', 'mkv', 'wmv'])

    if uploaded_video is not None:
        file_path = os.path.join(temp_dir, uploaded_video.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        if st.button('开始预处理'):
            play_video(file_path)  # 可选步骤，仅用于确认视频是否上传正确，可以注释掉

            frame_dir = './video_frame_picture'
            os.makedirs(frame_dir, exist_ok=True)
            extract_frames(file_path, frame_dir)
            st.success(f'Frames extracted to {frame_dir}')

            delete_similar_images(frame_dir, threshold=1200)
            st.success('Similar images deleted.')

            process_images(frame_dir)
            st.success('Images processed.')

            filter_images(frame_dir, threshold=0.001)
            st.success('Low complexity images filtered.')

            # 如果不需要查看视频，建议注释掉play_video调用，避免在实际应用中无谓地播放视频。
