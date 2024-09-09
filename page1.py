import streamlit as st
import tensorflow as tf
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# # 加载预训练的MobileNetV2模型
# model = MobileNetV2(weights='imagenet')

# 加载预训练的ResNet50模型
model = ResNet50(weights='imagenet')

def run_image_prediction_page():
    st.title("图像分类器")

    # 图片上传
    uploaded_file = st.file_uploader("选择一张图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 打开并显示图像
        img = Image.open(uploaded_file)
        st.image(img, caption="上传的图片", use_column_width=True)

        # 对图像进行预处理
        img = img.resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # 进行预测
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=3)[0]

        # 显示预测结果
        st.write("预测结果：")
        for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
            st.write(f"{i + 1}: {label} ({score:.2%})")
