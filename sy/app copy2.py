import sys
import os
import json
import streamlit as st
import pandas as pd
import re
import io
import time
import sqlite3
import torch
from pathlib import Path
import warnings
import logging

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ==========================================
# 跨文件夹导包路径配置
# ==========================================
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

paths_to_add = [
    str(project_root)  
]

for p in paths_to_add:
    if p not in sys.path:
        sys.path.insert(0, p)

from mnx.tuomin_api import api_desensitize_df
from mnx.bm25_algo.bm25_api import api_bm25_df
from mnx.classification.model_nn_train_3layers_better import api_predict_categories, api_train_model
from mnx.beautifulLook.fangshanmap_visualization import api_fangshanmap_visualization
from mnx.accountLedger.Ledger import api_account_ledger, CATEGORY_MAP,REVERSE_CATEGORY_MAP,init_db,save_to_db
from mnx.extract.extract_and_mask_api import api_extract_and_mask

def extract_wage_info(texts):
    """
    针对第19类投诉内容进行关键特征提取。
    包含：欠薪主体、涉事负责人、金额、项目名称、地点、时间、人数、合同情况。
    """
    results = []
    for text in texts:
        # 1. 涉及金额
        money_match = re.search(r'(\d+([.][\d]+)?\s*[万千百十]?[元块])', text)
        money_val = money_match.group(0) if money_match else "未明确提及"
        
        # 2. 疑似欠薪主体/单位
        company_match = re.search(r'([\u4e00-\u9fa5]{2,20}(?:公司|厂|项目部|合作社|中心|饭店|商场|劳务))', text)
        company_val = company_match.group(0) if company_match else "需人工核实"
        
        # 3. 涉事负责人
        boss_match = re.search(r'(老板|包工头|负责人|叫|姓)([\u4e00-\u9fa5]{1,3})(老板|师傅|先生|女士)?', text)
        boss_val = boss_match.group(2) if boss_match else "未知"

        # 4. 工程项目名称
        project_match = re.search(r'([\u4e00-\u9fa5]{2,20}(?:项目|工程|标段|小区|大厦))', text)
        project_val = project_match.group(0) if project_match else "未明确提及"

        # 5. 工程地点
        location_match = re.search(r'([\u4e00-\u9fa5]{2,15}(?:省|市|区|县|镇|街道|村|工地|路\d+号))', text)
        location_val = location_match.group(0) if location_match else "未明确提及"

        # 6. 时间节点 (开工/拖欠时间)
        date_match = re.search(r'(\d{4}年\d{1,2}月(?:\d{1,2}日)?|\d{2,4}[-/]\d{1,2}[-/]\d{1,2})', text)
        date_val = date_match.group(0) if date_match else "未明确提及"

        # 7. 工人人数
        workers_match = re.search(r'([一二三四五六七八九十百千万\d]+)\s*(?:个|名)?(?:人|工人|农民工|兄弟)', text)
        workers_val = workers_match.group(0) if workers_match else "未知 / 疑似单人"

        # 8. 劳动合同签订情况
        if re.search(r'(没签|未签|没有签|拒签|口头).*合同', text) or re.search(r'合同.*(没签|未签|没有签|拒签)', text):
            contract_val = "未签订 (有风险)"
        elif re.search(r'(签了|签订|签署|有).*合同', text) or re.search(r'合同.*(签了|签订|签署)', text):
            contract_val = "已签订"
        else:
            contract_val = "线索未提及"

        results.append({
            "原投诉内容": text,
            "疑似欠薪主体": company_val,
            "涉事负责人": boss_val,
            "涉及金额": money_val,
            "工程项目名称": project_val,
            "工程地点": location_val,
            "时间节点": date_val,
            "工人人数": workers_val,
            "劳动合同情况": contract_val
        })
    return pd.DataFrame(results)

# ==========================================
# 模块3：UI 与 交互流转
# ==========================================
def show_app():
    st.set_page_config(page_title="12345涉检线索智能筛查系统", layout="wide")
    init_db()

    # ==========================================
    # 极简科技感全局 CSS 样式
    # ==========================================
    st.markdown("""
            <style>
                /* 全局背景与字体 - 增强科技感 */
                .stApp {
                    background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0f172a 100%);
                    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                    color: #e2e8f0;
                }
                
                /* 隐藏默认装饰元素 - 完全移除白色区域 */
                #MainMenu {visibility: hidden !important;}
                .stDeployButton {display:none !important;}
                footer {visibility: hidden !important;}
                #stDecoration {display:none !important;}
                header {visibility: hidden !important;}  /* 隐藏顶部白色header */
                .stApp > header {display: none !important;}  /* 双重保险隐藏header */
                
                /* 主标题样式 - 增强科技感 */
                .main-title {
                    color: #00d4ff;
                    font-size: 2.4rem;
                    font-weight: 700;
                    text-shadow: 0 0 10px rgba(0, 212, 255, 0.5), 0 2px 4px rgba(0,0,0,0.3);
                    margin-bottom: 0.5rem;
                    background: linear-gradient(90deg, #00d4ff, #0080ff);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }
                
                /* 顶部导航栏容器 - 更现代的玻璃态效果 */
                .nav-container {
                    background: rgba(10, 14, 26, 0.7);
                    backdrop-filter: blur(15px);
                    border-radius: 12px;
                    padding: 1rem 1.5rem;
                    margin-bottom: 1.5rem;
                    border: 1px solid rgba(0, 212, 255, 0.2);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
                }
                
                /* Tab 按钮样式 - 霓虹科技风格 */
                .stTabs [data-baseweb="tab-list"] {
                    gap: 10px;
                    background: rgba(10, 14, 26, 0.5);
                    padding: 8px;
                    border-radius: 12px;
                }
                
                .stTabs [data-baseweb="tab"] {
                    height: 50px;
                    white-space: pre-wrap;
                    background: rgba(20, 25, 45, 0.8);
                    border-radius: 8px;
                    color: #94a3b8;
                    font-weight: 600;
                    padding: 0 1.5rem;
                    transition: all 0.3s ease;
                    border: 1px solid rgba(100, 100, 150, 0.2);
                }
                
                .stTabs [aria-selected="true"] {
                    background: linear-gradient(135deg, #0066ff 0%, #00ccff 100%);
                    color: white;
                    box-shadow: 0 0 20px rgba(0, 102, 255, 0.5), 0 4px 12px rgba(0, 0, 0, 0.3);
                    border: 1px solid rgba(0, 204, 255, 0.5);
                }
                
                .stTabs [data-baseweb="tab"]:hover {
                    background: rgba(30, 40, 70, 0.9);
                    color: #00d4ff;
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0, 212, 255, 0.2);
                }
                
                /* 内容区域卡片 - 增强深度感 */
                .content-card {
                    background: rgba(10, 14, 26, 0.6);
                    backdrop-filter: blur(10px);
                    border-radius: 12px;
                    padding: 1.5rem;
                    border: 1px solid rgba(0, 212, 255, 0.1);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
                    margin-bottom: 1.5rem;
                    transition: all 0.3s ease;
                }
                
                .content-card:hover {
                    border-color: rgba(0, 212, 255, 0.3);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), 0 0 15px rgba(0, 212, 255, 0.1);
                }
                
                /* 数据表格样式 */
                .stDataFrame {
                    background: rgba(10, 14, 26, 0.8);
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid rgba(0, 212, 255, 0.1);
                }
                
                /* 按钮样式增强 - 科技感按钮 */
                .stButton > button {
                    border-radius: 8px;
                    font-weight: 600;
                    transition: all 0.3s ease;
                    border: none;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                
                .stButton > button[kind="primary"] {
                    background: linear-gradient(135deg, #0066ff 0%, #00ccff 100%);
                    box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4);
                    color: white;
                }
                
                .stButton > button[kind="primary"]:hover {
                    box-shadow: 0 6px 20px rgba(0, 102, 255, 0.6), 0 0 15px rgba(0, 204, 255, 0.3);
                    transform: translateY(-2px);
                }
                
                .stButton > button[kind="secondary"] {
                    background: rgba(20, 25, 45, 0.8);
                    color: #e2e8f0;
                    border: 1px solid rgba(100, 100, 150, 0.3);
                }
                
                .stButton > button[kind="secondary"]:hover {
                    background: rgba(30, 40, 70, 0.9);
                    border-color: rgba(0, 212, 255, 0.5);
                    transform: translateY(-2px);
                }
                
                /* 文件上传器样式 */
                .uploadedFile {
                    background: rgba(10, 14, 26, 0.6);
                    border-radius: 8px;
                    border: 2px dashed rgba(0, 212, 255, 0.3);
                }
                
                /* Selectbox 样式 */
                .stSelectbox > div > div {
                    background: rgba(10, 14, 26, 0.8);
                    border-radius: 8px;
                    border: 1px solid rgba(0, 212, 255, 0.2);
                    color: #e2e8f0;
                }
                
                /* 文本颜色统一 - 确保没有白色背景透出 */
                h1, h2, h3, h4, h5, h6 {
                    color: #e2e8f0 !important;
                }
                
                p, span, div {
                    color: #cbd5e1 !important;
                }
                
                /* 分隔线 */
                hr {
                    border-color: rgba(0, 212, 255, 0.2) !important;
                }
                
                /* Expander 样式 */
                .streamlit-expanderHeader {
                    background: rgba(10, 14, 26, 0.6);
                    border-radius: 8px;
                    border: 1px solid rgba(0, 212, 255, 0.2);
                    color: #e2e8f0 !important;
                }
                
                /* 进度条样式 */
                .stProgress > div > div > div > div {
                    background: linear-gradient(90deg, #0066ff 0%, #00ccff 100%);
                }
                
                /* 警告/成功信息框 */
                .stAlert {
                    border-radius: 8px;
                    border: none;
                    background: rgba(10, 14, 26, 0.8) !important;
                    border-left: 4px solid #00d4ff !important;
                }
                
                /* 右侧面板特殊样式 - 更立体 */
                .wage-panel {
                    background: linear-gradient(135deg, rgba(10, 14, 26, 0.9) 0%, rgba(20, 25, 45, 0.8) 100%);
                    border-radius: 12px;
                    padding: 1.5rem;
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05);
                }
                
                /* 滚动条样式 - 增强科技感 */
                ::-webkit-scrollbar {
                    width: 8px;
                    height: 8px;
                }
                
                ::-webkit-scrollbar-track {
                    background: rgba(10, 14, 26, 0.5);
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb {
                    background: linear-gradient(135deg, #0066ff, #00ccff);
                    border-radius: 4px;
                }
                
                ::-webkit-scrollbar-thumb:hover {
                    background: linear-gradient(135deg, #00ccff, #0066ff);
                }
            </style>
        """, unsafe_allow_html=True)

    if "show_wage_panel" not in st.session_state:
        st.session_state.show_wage_panel = False
    if "db_refresh_trigger" not in st.session_state:
        st.session_state.db_refresh_trigger = True
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "台账查看"

    # ---------------- 顶部标题与导航工具栏 ----------------
    st.markdown('<div class="main-title">⚖️ 12345 涉检线索智能筛查系统</div>', unsafe_allow_html=True)

    # 顶部导航 Tabs
    tab_names = ["📋 台账查看", "📂 数据导入与预测", "🛠️ 模型重训", "💰 拖欠工资分析", "可视化显示"]
    tabs = st.tabs(tab_names)
    
    
    # ---------------- Tab 1: 台账查看 ----------------
    with tabs[0]:
        api_account_ledger()
    # ---------------- Tab 2: 数据导入与预测流水线 ----------------
    with tabs[1]:
        st.session_state.active_tab = "数据导入与预测"

        # 系统状态显示（右上角）
        status_col = st.columns([1])
        with status_col[0]:
            # st.markdown('<div class="content-card">', unsafe_allow_html=True)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.success(f"**GPU 加速** 🚀\n{gpu_name}")
            else:
                st.warning("**CPU 模式** 🐌")
            # st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("📂 数据导入与预测流水线")
        
        uploaded_file = st.file_uploader("📤 导入12345原始待预测数据", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            if "raw_df" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.raw_df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.raw_df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.desensitized_df = None

            pred_text_col = st.selectbox("🎯 指定待预测数据中的 '特征文本' 列", st.session_state.raw_df.columns, index=0)

            if st.session_state.get("uploaded_filename") is not None:
                st.markdown("---")
                st.subheader("🛠️ 第一步：执行脱敏")
                if st.button("🛡️ 对摘要执行专业脱敏", type="primary"):
                    with st.spinner("正在清洗摘要数据..."):
                        try:
                            progress_bar = st.progress(0, text="脱敏进度: 0%")
                            total_count = len(st.session_state.raw_df)
                            processed_count = [0]

                            def update_progress(step):
                                processed_count[0] += step
                                progress = min(processed_count[0] / total_count, 1.0)
                                progress_bar.progress(progress, text=f"脱敏进度: {int(progress * 100)}%")

                            df_desen = api_extract_and_mask(st.session_state.raw_df, text_column=pred_text_col, progress_callback=update_progress)
                            st.session_state.desensitized_df = df_desen
                            st.success("脱敏清洗完成！")
                        except Exception as e:
                            st.error(f"脱敏过程出错: {str(e)}")

                if st.session_state.get("desensitized_df") is not None:
                    # st.markdown("---")
                    # st.subheader("🧠 第二步：真实的AI大模型推理")
                    # if st.button("🚀 调用 3层MLP 引擎预测并入库"):
                    #     with st.spinner("正在将文本投入大模型获取特征向量并进行推理，请耐心等待..."):
                    #         df_to_predict = st.session_state.desensitized_df.copy()
                    #         texts_to_predict = df_to_predict[pred_text_col].tolist()

                    #         try:
                    #             pred_labels_str = api_predict_categories(texts_to_predict)
                    #             predictions = []
                    #             for label_str in pred_labels_str:
                    #                 cat_id = REVERSE_CATEGORY_MAP.get(label_str, 0)
                    #                 predictions.append(cat_id)

                    #             df_to_predict["预测类别"] = predictions
                    #             df_to_predict["处理状态"] = "已筛查"
                    #             df_to_predict["人工复核"] = "待复核"

                    #             save_to_db(df_to_predict, pred_text_col)
                    #             st.session_state.db_refresh_trigger = True
                    #             st.success("🎉 AI 预测并落库成功！请在【台账查看】标签页查看结果。")
                    #         except Exception as e:
                    #             st.error(f"模型推理失败: {e}")
                    if st.session_state.get("desensitized_df") is not None:
                        st.markdown("---")
                        st.subheader("🧠 第二步：真实的AI大模型推理")
                        if st.button("🚀 调用 3层MLP 引擎预测并入库"):
                            df_to_predict = st.session_state.desensitized_df.copy()
                            texts_to_predict = df_to_predict[pred_text_col].tolist()

                            pred_bar = st.progress(0, text="准备推理...")
                            try:
                                def pred_progress_cb(percent, msg):
                                    pred_bar.progress(percent / 100.0, text=msg)

                                pred_labels_str = api_predict_categories(texts_to_predict, progress_cb=pred_progress_cb)
                                predictions = []
                                for label_str in pred_labels_str:
                                    cat_id = REVERSE_CATEGORY_MAP.get(label_str, 0)
                                    predictions.append(cat_id)

                                df_to_predict["预测类别"] = predictions
                                df_to_predict["处理状态"] = "已筛查"
                                df_to_predict["人工复核"] = "待复核"

                                save_to_db(df_to_predict, pred_text_col)
                                st.session_state.db_refresh_trigger = True
                                pred_bar.empty()
                                st.success("🎉 AI 预测并落库成功！请在【台账查看】标签页查看结果。")
                            except Exception as e:
                                pred_bar.empty()
                                st.error(f"模型推理失败: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Tab 3: 模型在线重训 ----------------
    with tabs[2]:
        st.session_state.active_tab = "模型重训"

        # 系统状态显示（右下角）
        status_col = st.columns([1])
        with status_col[0]:
            # st.markdown('<div class="content-card">', unsafe_allow_html=True)
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.success(f"**GPU 加速** 🚀\n{gpu_name}")
            else:
                st.warning("**CPU 模式** 🐌")
            # st.markdown('</div>', unsafe_allow_html=True)

        st.subheader("🛠️ AI 模型在线重训")
        st.info("通过上传历史数据和自定义标签字典，系统将重训并动态适配新的法定领域。")
        
        train_file = st.file_uploader("1. 导入训练集 (需含特征与标签ID)", type=["csv", "xlsx"], key="train_data")
        map_file = st.file_uploader("2. 导入标签映射表 (标签ID <-> 中文名称)", type=["csv", "xlsx"], key="map_data")

        if train_file and map_file:
            try:
                df_t = pd.read_csv(train_file) if train_file.name.endswith('.csv') else pd.read_excel(train_file)
                df_m = pd.read_csv(map_file) if map_file.name.endswith('.csv') else pd.read_excel(map_file)

                t_text_col = st.selectbox("🎯 指定训练集中的 '特征文本' 列", df_t.columns, index=0)
                t_label_col = st.selectbox("🎯 指定训练集中的 '标签序号' 列", df_t.columns, index=min(1, len(df_t.columns) - 1))

                m_id_col = st.selectbox("📚 指定映射表中的 '标签序号' 列", df_m.columns, index=0)
                m_name_col = st.selectbox("📚 指定映射表中的 '中文名称' 列", df_m.columns, index=min(1, len(df_m.columns) - 1))

                if st.button("🚀 开始训练最新的 3层神经网络", type="primary"):
                    train_bar = st.progress(0, text="准备开始训练...")
                    def st_progress_cb(percent, msg):
                        train_bar.progress(percent / 100.0, text=msg)

                    try:
                        api_train_model(df_t, df_m, t_text_col, t_label_col, m_id_col, m_name_col, progress_cb=st_progress_cb)
                        st.success("✅ 模型重训圆满成功！新标签库与新推理引擎已自动上线。")
                        time.sleep(2)
                        st.rerun()
                    except Exception as e:
                        st.error(f"训练中断: {str(e)}")
            except Exception as e:
                st.error("读取上传文件失败，请确保格式正确。")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Tab 4: 拖欠工资专项分析 ----------------
    with tabs[3]:
        st.session_state.active_tab = "拖欠工资分析"
        # st.markdown('<div class="wage-panel">', unsafe_allow_html=True)
        st.markdown("### 👷 拖欠工资专项提取")
        st.info("自动过滤数据库中的**第19类**数据，并提取全维度欠薪要素。")
        
        if "db_data" in st.session_state and not st.session_state.db_data.empty:
            wage_target_df = st.session_state.db_data[st.session_state.db_data['pred_category'].astype(str) == '19']
            
            if wage_target_df.empty:
                st.warning("⚠️ 未在数据库中发现第19类(拖欠工资)的线索数据。")
            else:
                st.success(f"已锁定 **{len(wage_target_df)}** 条拖欠工资线索数据。")
                
                if st.button("⚡ 一键提取全维度特征", type="primary", use_container_width=True):
                    with st.spinner("正在逐条深度解析文本特征..."):
                        extracted_df = extract_wage_info(wage_target_df['content'].tolist())
                        st.session_state.wage_analysis_res = extracted_df
                        
                if "wage_analysis_res" in st.session_state:
                    st.dataframe(st.session_state.wage_analysis_res, use_container_width=True, hide_index=True)
                    
                    wage_buffer = io.BytesIO()
                    with pd.ExcelWriter(wage_buffer, engine='xlsxwriter') as writer:
                        st.session_state.wage_analysis_res.to_excel(writer, index=False, sheet_name='拖欠工资全维度要素')
                        
                    st.download_button(
                        label="⬇️ 导出完整专项分析报表 (Excel)", 
                        data=wage_buffer.getvalue(),
                        file_name="拖欠工资线索_深度特征分析.xlsx", 
                        mime="application/vnd.ms-excel",
                        use_container_width=True
                    )
        else:
             st.warning("请先在【数据导入与预测】标签页完成数据导入与推理。")
        
        # st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- Tab 5: 可视化分析 ----------------
    with tabs[4]:
        st.session_state.active_tab = "可视化分析"
        
        # 初始化地图选择的session_state
        if "selected_township" not in st.session_state:
            st.session_state.selected_township = None
        
        api_fangshanmap_visualization()
        



if __name__ == "__main__":
    show_app()