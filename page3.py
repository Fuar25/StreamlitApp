import cv2
import streamlit as st
from PIL import Image
import os

def run_video_frame_extraction_page():
    # 设置页面标题
    st.title("视频抽帧工具")

    # 文件上传
    video_file = st.file_uploader("上传视频文件", type=["mp4", "avi"])

    # 抽帧频率设置
    frame_interval = st.slider("选择抽帧频率（每几帧抽取一帧）", 1, 60, 1)

    # 选择保存路径
    output_folder = st.text_input("输入保存图片的文件夹路径", "")

    # 创建保存文件夹
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 视频处理按钮
    if st.button("开始抽帧"):
        if video_file is not None and output_folder:
            # 将上传的视频文件保存到临时位置
            temp_video_path = os.path.join(output_folder, 'temp_video.mp4')
            with open(temp_video_path, 'wb') as f:
                f.write(video_file.read())

            # 打开视频文件
            cap = cv2.VideoCapture(temp_video_path)

            # 初始化帧计数器
            frame_count = 0

            # 开始读取视频帧
            while(cap.isOpened()):
                ret, frame = cap.read()
                if not ret:
                    break

                # 根据设定的间隔保存帧
                if frame_count % frame_interval == 0:
                    output_path = os.path.join(output_folder, f'frame_{frame_count}.jpg')
                    cv2.imwrite(output_path, frame)

                frame_count += 1

            # 释放资源
            cap.release()

            st.success(f"抽帧完成！共抽取了{frame_count // frame_interval}帧。")

            # 显示部分抽帧结果
            for i in range(5):  # 显示前5张图片作为示例
                img_path = os.path.join(output_folder, f'frame_{i * frame_interval}.jpg')
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, caption=f'Frame {i * frame_interval}', use_column_width=True)

        elif not output_folder:
            st.warning("请输入保存图片的文件夹路径！")
        else:
            st.warning("请先上传视频文件！")

if __name__ == "__main__":
    run_video_frame_extraction_page()
