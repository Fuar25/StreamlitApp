import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def parse_log_file(log_content):
    """
    解析日志文件内容，提取训练历史记录
    :param log_content: 日志文件的内容
    :return: DataFrame 包含训练历史记录
    """
    lines = log_content.decode().split('\n')
    data = []

    # 跳过第一行标题行
    skip_header = True
    for line in lines:
        if skip_header:
            skip_header = False
            continue
        parts = line.split(',')
        if len(parts) >= 3:
            try:
                epoch = int(parts[0])
                loss = float(parts[1])
                acc = float(parts[2])

                # 检查是否包含验证数据
                if len(parts) >= 5:
                    val_loss = float(parts[3])
                    val_acc = float(parts[4])
                    data.append([epoch, loss, val_loss, acc, val_acc])
                else:
                    data.append([epoch, loss, None, acc, None])
            except ValueError:
                continue

    df = pd.DataFrame(data, columns=['epoch', 'loss', 'val_loss', 'accuracy', 'val_accuracy'])
    return df

def plot_learning_curves(history):
    """
    绘制模型训练的学习曲线
    :param history: 模型训练历史记录，包含 loss 和 accuracy 等指标
    """
    # 提取训练和验证的损失和准确率
    train_loss = history['loss']
    val_loss = history.get('val_loss', [])
    train_acc = history['accuracy']
    val_acc = history.get('val_accuracy', [])

    # 创建一个新的图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 绘制损失曲线
    ax1.plot(train_loss, label='Training Loss')
    if val_loss.any():
        ax1.plot(val_loss, label='Validation Loss', linestyle='--')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # 绘制准确率曲线
    ax2.plot(train_acc, label='Training Accuracy')
    if val_acc.any():
        ax2.plot(val_acc, label='Validation Accuracy', linestyle='--')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    # 显示图形
    plt.tight_layout()
    st.pyplot(fig)

def run_model_visualization_page():
    st.title('模型训练效果可视化')

    # 上传日志文件
    uploaded_file = st.file_uploader("上传日志文件", type=['txt', 'log'])

    if st.button('开始可视化'):
        if uploaded_file is not None:
            try:
                # 解析日志文件内容
                history = parse_log_file(uploaded_file.read())

                # 绘制学习曲线
                plot_learning_curves(history)
            except Exception as e:
                st.error(f"解析或绘制历史记录时发生错误：{e}")
        else:
            st.warning("请上传有效的日志文件。")

if __name__ == '__main__':
    run_model_visualization_page()
