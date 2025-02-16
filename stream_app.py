import streamlit as st
import pandas as pd
import pickle
import os
import shap

# 修改模型路径
model_path = os.path.join(os.getcwd(), 'random_forest_model.pkl')

# 加载模型
with open(model_path, 'rb') as file:
    model = pickle.load(file)

st.set_page_config(page_title="卵巢高反应预测计算器", layout="wide")
st.title("卵巢高反应风险预测工具")

# 侧边栏输入新增BMI
with st.sidebar:
    st.header("患者参数输入")
    st.subheader("卵巢储备指标")    
    amh = st.slider("AMH (ng/mL)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
    afc = st.slider("AFC (个)", min_value=0, max_value=40, value=15, step=1)
    fsh = st.slider("基础FSH (IU/L)", min_value=1.0, max_value=20.0, value=8.0, step=0.1)
    
    st.subheader("基础特征")
    age = st.slider("年龄 (years old)", min_value=18, max_value=50, value=30)
    bmi = st.slider("BMI (kg/m²)", min_value=15.0, max_value=40.0, value=22.0, step=0.1)  # 新增BMI输入

# 调整输入数据顺序（关键！必须与训练数据顺序一致）
input_data = pd.DataFrame({
    'AMH': [amh],
    'AFC': [afc],
    'FSH': [fsh],
    'age': [age],
    'bmi': [bmi]  # 新增BMI
})[["AMH", "AFC", "FSH", "age", "bmi"]]  # 确保列顺序与训练数据一致

# 预测与解释
col1, col2 = st.columns([1, 2])

with col1:
    if st.button("开始风险评估"):
        prob = model.predict_proba(input_data)[0][1]
        # ...保持原有风险计算逻辑...

with col2:
    if 'prob' in locals():
        # SHAP解释需要调整索引
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)
        
        # 使用二维索引 [1][0] 获取正类解释
        fig = shap.plots.waterfall(shap_values[1][0], max_display=5)  # 显示前5个重要特征
        st.pyplot(fig)
        
        # 更新特征说明
        st.markdown("""
        **特征说明**:
        - 正值增加风险，负值降低风险
        - AMH/AFC仍是主要预测因子
        - BMI呈现U型影响（过低/过高均增加风险）
        """)

# 注意事项
st.markdown("---")
st.warning("""
**使用限制**:
1. 适用于未接受过卵巢手术的患者
2. 最终决策需结合临床判断
""")