import streamlit as st
from page1 import run_image_prediction_page
from page2 import run_video_frame_prediction_page
from page3 import run_video_frame_extraction_page
from page4 import run_image_selection_page
from page5 import run_model_visualization_page
from page6 import run_image_parameter_page


# Streamlit 应用标题
st.sidebar.title("选择功能")
page = st.sidebar.selectbox("选择页面", ["图片预测", "视频预测", "视频抽帧", "逐帧筛选", "模型训练", "图片参数"])

if page == "图片预测":
    run_image_prediction_page()
elif page == "视频预测":
    run_video_frame_prediction_page()
elif page ==  "视频抽帧":
    run_video_frame_extraction_page()
elif page ==  "逐帧筛选":
    run_image_selection_page()
elif page ==  "图片参数":
    run_image_parameter_page()
elif page ==  "模型训练":
    run_model_visualization_page()

