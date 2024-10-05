import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np

# 加载已训练好的模型
model = tf.keras.models.load_model('DemoWithoutFT.keras')

# 定义预测函数
def predict(image):
    # 预处理图像
    image = image.resize((160, 160))  # 调整图像大小
    image = np.array(image) / 255.0  # 归一化图像
    image = np.expand_dims(image, axis=0)  # 扩展维度以匹配模型输入

    # 进行预测
    predictions = model.predict(image)
    return predictions

# Streamlit 应用程序
def main():
    st.title('图像分类器')

    # 文件上传
    uploaded_file = st.file_uploader("选择一张图片", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file)
        st.image(image, caption='上传的图片', use_column_width=True)

        # 进行预测
        prediction = predict(image)
        probability = tf.nn.sigmoid(prediction)
        st.markdown('---')
        st.write(f'预测结果:{prediction}')
        st.write(f'预测结果:{probability}')
        st.markdown('---')
        if probability > 0.5:
            st.write(f'预测结果: 狗 ')
        else:
            st.write(f'预测结果: 猫 ')

if __name__ == '__main__':
    main()
