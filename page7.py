import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# 加载模型
def load_model():
    model = keras.models.load_model('DemoWithFT.keras')
    return model

# 预处理图像
def preprocess_image(image):
    # 将图像转换为灰度图
    image = image.convert('L')
    # 调整大小到28x28
    image = image.resize((28, 28))
    # 归一化像素值
    image = np.array(image) / 255.0
    # 增加维度以匹配模型输入要求
    image = np.expand_dims(image, axis=0)
    # 添加颜色通道维度
    image = np.expand_dims(image, axis=-1)
    return image

# 主程序
def main():
    st.title("图片识别")

    # 文件上传
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "png"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图片', width=200)

        # 预处理图片
        processed_image = preprocess_image(image)

        # 加载模型并预测
        model = load_model()
        prediction = model.predict(processed_image)

        # 获取预测结果
        predicted_number = np.argmax(prediction)
        confidence = np.max(prediction)

        # 显示预测结果
        st.write(f'预测数字为: {predicted_number}')
        st.write(f'置信度: {confidence * 100:.2f}%')

if __name__ == "__main__":
    main()
