import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# 加载预训练的MobileNetV2模型
model = MobileNetV2(weights='imagenet')

def run_video_frame_prediction_page():
    st.title("视频分类器")

    # 视频上传
    uploaded_video = st.file_uploader("选择一个视频文件...", type=["mp4", "avi"])

    if uploaded_video is not None:
        # 保存视频文件
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_video.getbuffer())

        # 读取视频
        video = cv2.VideoCapture("temp_video.mp4")

        # 检查视频是否成功打开
        if not video.isOpened():
            st.error("无法打开视频文件，请检查文件格式。")
        else:
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = video.get(cv2.CAP_PROP_FPS)
            st.write(f"视频总帧数: {frame_count}, FPS: {fps}")

            # 设置抽帧间隔（每秒抽取一帧）
            interval = int(fps)

            # 抽帧并进行预测
            frame_index = 0
            while True:
                ret, frame = video.read()
                if not ret:
                    break

                if frame_index % interval == 0:
                    # 将 OpenCV 的 BGR 图像转换为 RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)

                    # 显示抽帧图像
                    st.image(img, caption=f"第 {frame_index} 帧", use_column_width=True)

                    # 对图像进行预处理
                    img = img.resize((224, 224))
                    img_array = image.img_to_array(img)
                    img_array = np.expand_dims(img_array, axis=0)
                    img_array = preprocess_input(img_array)

                    # 进行预测
                    predictions = model.predict(img_array)
                    decoded_predictions = decode_predictions(predictions, top=3)[0]

                    # 显示预测结果
                    st.write(f"第 {frame_index} 帧的预测结果：")
                    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
                        st.write(f"{i + 1}: {label} ({score:.2%})")

                frame_index += 1

            # 释放视频资源
            video.release()
