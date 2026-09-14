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
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
from mnx.duplicatedeal.duplicatedeal_api import api_duplicatedeal_api
from mnx.duplicatedeal.similarText_api import multi2one,one2multi
# 公益/私益诉讼分类模块（关键词匹配）
from mnx.classification.public_private_classifier import api_classify_public_private, filter_public_interest
# 统一类别映射管理器
from mnx.classification.category_mapper import get_category_id, validate_and_fix_category_name


@st.dialog("🚪 退出系统", width="small")
def logout_confirm_dialog():
    """退出登录确认弹窗"""
    st.write("确定要退出登录吗？")
    
    col_confirm, col_cancel = st.columns(2)
    with col_confirm:
        if st.button("✅ 是，退出登录", use_container_width=True, type="primary"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            qp = st.query_params
            qp["page"] = "login"
            qp["logged_in"] = "false"
            st.session_state.active_menu = "📋 台账查看"
            st.session_state.show_logout_dialog = False
            st.rerun()
    with col_cancel:
        if st.button("❌ 取消", use_container_width=True, type="secondary"):
            st.session_state.show_logout_dialog = False
            st.rerun()

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

                /* ========== 下拉菜单样式（更多功能）========== */
                
                .dropdown-container {
                    position: relative;
                    display: inline-block;
                }

                .dropdown-content {
                    display: none;
                    position: absolute;
                    top: 100%;
                    left: 0;
                    background: rgba(15, 23, 42, 0.95);
                    backdrop-filter: blur(20px);
                    min-width: 200px;
                    border-radius: 8px;
                    border: 1px solid rgba(0, 212, 255, 0.3);
                    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.5), 0 0 15px rgba(0, 212, 255, 0.1);
                    z-index: 9999;
                    overflow: hidden;
                }

                .dropdown-container:hover .dropdown-content {
                    display: block;
                }

                .dropdown-item {
                    padding: 12px 18px;
                    color: #cbd5e1;
                    font-size: 14px;
                    font-weight: 500;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    border-bottom: 1px solid rgba(0, 212, 255, 0.1);
                    text-decoration: none;
                    display: block;
                }

                .dropdown-item:hover {
                    background: linear-gradient(90deg, rgba(0, 102, 255, 0.3) 0%, rgba(0, 204, 255, 0.1) 100%);
                    color: #00d4ff;
                    padding-left: 24px;
                }

                .dropdown-item:last-child {
                    border-bottom: none;
                }
            </style>
        """, unsafe_allow_html=True)

    if "show_wage_panel" not in st.session_state:
        st.session_state.show_wage_panel = False
    if "db_refresh_trigger" not in st.session_state:
        st.session_state.db_refresh_trigger = True
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "台账查看"
    if "selected_township" not in st.session_state:
        st.session_state.selected_township = None
    if "active_menu" not in st.session_state:
        st.session_state.active_menu = "📋 台账查看"

    # ---------------- 顶部标题与导航工具栏 ----------------
    st.markdown('<div class="main-title">⚖️ 12345 涉检线索智能筛查系统</div>', unsafe_allow_html=True)

    # 顶部导航 Tabs
    menu_options = ["📋 台账查看", "📂 数据导入与预测", "🛠️ 模型重训"]
    cols_main = st.columns(len(menu_options) + 2) 
    
    for i, option in enumerate(menu_options):
        with cols_main[i]:
            is_active = st.session_state.active_menu == option
            if st.button(option, key=f"menu_{i}", use_container_width=True,
                        type="primary" if is_active else "secondary"):
                st.session_state.active_menu = option
                st.rerun()

    with cols_main[len(menu_options)]:
        # 判断当前是否选中了某个子项
        active_sub = st.session_state.active_menu in ["💰 拖欠工资分析", "可视化显示","多对一分析","一对多分析","多级预警机制"]
        
        popover_label = "🔧 更多功能 ▾"

        with st.popover(popover_label, use_container_width=True):
            sub_options = ["💰 拖欠工资分析", "可视化显示", "多对一分析", "一对多分析","多级预警机制"]
            sub_labels = ["💰 拖欠工资专项分析", "🗺️ 地图可视化分析", "多对一分析", "一对多分析","多级预警机制"]
            
            for sub_opt, sub_label in zip(sub_options, sub_labels):
                is_sub_active = st.session_state.active_menu == sub_opt
                if st.button(
                    sub_label, 
                    key=f"sub_menu_{sub_opt}",
                    use_container_width=True,
                    type="primary" if is_sub_active else "secondary"
                ):
                    st.session_state.active_menu = sub_opt
                    st.rerun()

    with cols_main[len(menu_options) + 1]:
        if st.button("🚪 退出系统", use_container_width=True, type="secondary"):
            st.session_state.show_logout_dialog = True
            st.rerun()

    if st.session_state.get("show_logout_dialog", False):
        logout_confirm_dialog()
        # 弹窗关闭后清理标记（如果用户点击取消）
        if not st.session_state.get("show_logout_dialog", False):
            st.rerun()

    # ---------- 从 URL query_params 读取菜单选择 ----------
    query_params = st.query_params
    if 'menu' in query_params:
        menu_val = query_params['menu']
        if menu_val != st.session_state.get('active_menu'):
            st.session_state.active_menu = menu_val
            del query_params['menu']
            st.rerun()

    st.markdown("---")

    # ---------------- Tab 1: 台账查看 ----------------
    if st.session_state.active_menu == "📋 台账查看":
        api_account_ledger()
    # ---------------- Tab 2: 数据导入与预测流水线 ----------------
    elif st.session_state.active_menu == "📂 数据导入与预测":
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

            pred_text_col = st.selectbox("🎯 指定待预测数据中的 '特征文本' 列", st.session_state.raw_df.columns, index=0, key="pred_text_col_select")
            # ---- 多级预警处理：在去重操作之前 ----
            st.markdown("---")
            
            # 第一行: 选择文本列 和 日期列
            col_warn_text, col_warn_date = st.columns(2)
            with col_warn_text:
                warning_text_col = st.selectbox(
                    "⚠️ 多级预警: 选择待分析的文本列", 
                    st.session_state.raw_df.columns, 
                    index=0, 
                    key="warning_text_col_select"
                )
            with col_warn_date:
                warning_date_col = st.selectbox(
                    "📅 选择日期/时间列 (用于计算投诉间隔)", 
                    st.session_state.raw_df.columns, 
                    index=0, 
                    key="warning_date_col_select"
                )
            
            # 第二行: 执行按钮
            col_warning_btn = st.columns([1])[0]
            with col_warning_btn:
                if st.button("🔥 执行多级预警分析", type="primary", key="btn_warning_analysis"):
                    with st.spinner("🔥 正在进行多级预警相似度分析..."):
                        try:
                            warning_groups = run_warning_analysis(
                                df=st.session_state.raw_df.copy(),
                                text_column=warning_text_col,
                                similarity_threshold=0.9
                            )
                            # 将结果和选中的日期列名都存入 session_state
                            st.session_state["warning_analysis_result"] = warning_groups
                            st.session_state["warning_date_col"] = warning_date_col
                            st.session_state["warning_raw_df_snapshot"] = st.session_state.raw_df.copy()
                            group_count = len(warning_groups)
                            total_involved = sum(len(g) for g in warning_groups)
                            st.success(
                                f"✅ 多级预警分析完成！共发现 **{group_count}** 个相似组，"
                                f"涉及 **{total_involved}** 条记录"
                            )
                        except Exception as e:
                            st.error(f"⚠️ 分析失败: {str(e)}")




            # ---- 去重操作：用户确认列选择后点击按钮触发 ----
            dedup_done_key = f"dedup_done_{st.session_state.uploaded_filename}"

            # 显示去重按钮（仅当尚未去重时显示）
            if not st.session_state.get(dedup_done_key):
                st.info(f"📌 当前选中列: **{pred_text_col}** | 共 **{len(st.session_state.raw_df)}** 条原始数据")
                if st.button("🔄 执行重复投诉检测与去除", type="primary", key="btn_dedup"):
                    with st.spinner("🔄 正在检测并去除重复投诉..."):
                        try:
                            deduped_df, removed_count = api_duplicatedeal_api(
                                st.session_state.raw_df.copy(), 
                                text_column=pred_text_col
                            )
                            st.session_state.raw_df = deduped_df
                            st.session_state[dedup_done_key] = True
                            st.success(f"✅ **重复投诉去除成功！** 已去除 **{removed_count}** 条重复数据，剩余 **{len(deduped_df)}** 条有效数据")
                        except Exception as e:
                            st.error(f"⚠️ 去重失败: {str(e)}")
                            st.session_state[dedup_done_key] = "failed"

            # 去重完成后的状态提示
            elif st.session_state.get(dedup_done_key) == True:
                curr_count = len(st.session_state.raw_df)
                col_info = st.session_state.get(f"dedup_col_{dedup_done_key}", pred_text_col)
                st.info(f"✅ 已完成自动去重 (基于列 **'{col_info}'**) | 当前有效数据: **{curr_count}** 条")
                if st.button("🔄 重新选择列并去重", key="btn_redo_dedup"):
                    del st.session_state[dedup_done_key]
                    st.rerun()

            elif st.session_state.get(dedup_done_key) == "failed":
                st.error("⚠️ 上次去重失败，请检查数据后重试")
                if st.button("🔄 重试去重", key="btn_retry_dedup"):
                    del st.session_state[dedup_done_key]
                    st.rerun()

            # -------------------------------------------------------

            if st.session_state.get("uploaded_filename") is not None:
                st.markdown("---")
                
                # ==================== 公益/私益分类筛选 ====================
                st.subheader("🛠️ 第一步：公益/私益诉讼筛选")
                st.info("💡 通过关键词匹配自动识别并剔除私益投诉，保留公益诉讼数据")
                
                if st.button("🔍 执行公益/私益分类（关键词匹配）", type="primary", key="classify_btn"):
                    with st.spinner("正在分析投诉内容，区分公益与私益诉讼..."):
                        try:
                            # 底层：处理全部数据进行分类（移除num_samples限制）
                            classified_df = api_classify_public_private(
                                st.session_state.raw_df.copy(),
                                text_column=pred_text_col
                                # 移除 num_samples 参数，处理全部数据
                            )
                            st.session_state.classified_df = classified_df
                            
                            # UI显示：基于实际分类结果统计
                            total_count = len(classified_df)
                            public_count = len(classified_df[classified_df['诉讼类型'] == '公益'])
                            private_count = len(classified_df[classified_df['诉讼类型'] == '私益'])
                            unknown_count = len(classified_df[classified_df['诉讼类型'].isna() | (classified_df['诉讼类型'] == '')])
                            
                            st.success(f"✅ 分类完成！共分析 {total_count} 条数据")
                            st.info(f"📊 统计结果：公益 {public_count} 条 | 私益 {private_count} 条 | 未知 {unknown_count} 条")
                            
                            # 显示分类结果预览
                            display_cols = [pred_text_col, '诉讼类型', '置信度']
                            st.dataframe(classified_df.iloc[:20][display_cols], use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"⚠️ 分类失败: {str(e)}")
                
                # 如果已分类，显示过滤按钮
                if st.session_state.get("classified_df") is not None:
                    col_filter1, col_filter2 = st.columns([1, 3])
                    with col_filter1:
                        if st.button("🗑️ 过滤私益诉讼，保留公益数据", type="secondary", key="filter_btn"):
                            with st.spinner("正在过滤私益数据..."):
                                try:
                                    # 底层：基于实际分类结果进行过滤（移除num_samples限制）
                                    filtered_df = filter_public_interest(
                                        st.session_state.classified_df.copy()
                                        # 移除 num_samples 参数，过滤全部数据
                                    )
                                    st.session_state.filtered_df = filtered_df
                                    
                                    # UI显示：基于实际过滤结果统计
                                    total_count = len(st.session_state.classified_df)
                                    public_count_final = len(filtered_df)
                                    private_count_final = total_count - public_count_final
                                    
                                    st.success(f"✅ 过滤完成！保留了 {public_count_final} 条公益诉讼数据")
                                    st.info(f"📊 最终统计：公益 {public_count_final} 条 | 私益 {private_count_final} 条(已剔除)")
                                except Exception as e:
                                    st.error(f"⚠️ 过滤失败: {str(e)}")
                    
                    with col_filter2:
                        if st.session_state.get("filtered_df") is not None:
                            total_original = len(st.session_state.classified_df)
                            public_count_display = len(st.session_state.filtered_df)
                            st.info(f"当前共有 {public_count_display} 条公益诉讼数据(原始 {total_original} 条)")
                
                st.markdown("---")
                st.subheader("🛠️ 第二步：执行脱敏")
                if st.button("🛡️ 对摘要执行专业脱敏", type="primary"):
                    with st.spinner("正在清洗摘要数据..."):
                        try:
                            progress_bar = st.progress(0, text="脱敏进度: 0%")
                            # 如果有过滤后的数据，使用过滤后的；否则使用原始数据
                            df_to_desen = st.session_state.get("filtered_df", st.session_state.raw_df)
                            total_count = len(df_to_desen)
                            processed_count = [0]

                            def update_progress(step):
                                processed_count[0] += step
                                progress = min(processed_count[0] / total_count, 1.0)
                                progress_bar.progress(progress, text=f"脱敏进度: {int(progress * 100)}%")

                            df_desen = api_extract_and_mask(df_to_desen, text_column=pred_text_col, progress_callback=update_progress)
                            st.session_state.desensitized_df = df_desen
                            st.success("脱敏清洗完成！")
                        except Exception as e:
                            st.error(f"脱敏过程出错: {str(e)}")

                if st.session_state.get("desensitized_df") is not None:
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
                                unmapped_labels = []  # 记录无法映射的标签
                                
                                for label_str in pred_labels_str:
                                    # 使用统一的类别映射管理器进行验证和转换
                                    cat_id, validated_name = validate_and_fix_category_name(label_str, default_id=0)
                                    
                                    if cat_id == 0 and label_str != "非公益诉讼":
                                        # 如果返回默认值且不是原本的非公益诉讼，说明映射失败
                                        unmapped_labels.append(label_str)
                                    
                                    predictions.append(cat_id)
                                
                                # 如果有未映射的标签，输出警告信息帮助调试
                                if unmapped_labels:
                                    unique_unmapped = list(set(unmapped_labels))
                                    st.warning(f"⚠️ 发现 {len(unmapped_labels)} 条记录的预测标签无法映射到已知类别：")
                                    st.code(f"未识别的标签: {unique_unmapped[:10]}")  # 只显示前10个
                                    st.info("请检查模型训练时的类别名称是否与 category_map.json 一致")

                                df_to_predict["预测类别"] = predictions
                                df_to_predict["处理状态"] = "已筛查"
                                df_to_predict["人工复核"] = "待复核"
                                
                                # 打印统计信息用于调试
                                from collections import Counter
                                pred_counter = Counter(predictions)
                                st.info(f"📊 预测类别分布统计: {dict(pred_counter)}")

                                save_to_db(df_to_predict, pred_text_col)
                                st.session_state.db_refresh_trigger = True
                                pred_bar.empty()
                                st.success("🎉 AI 预测并落库成功！请在【台账查看】标签页查看结果。")
                            except Exception as e:
                                pred_bar.empty()
                                st.error(f"模型推理失败: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    # ---------------- Tab 3: 模型在线重训 ----------------
    elif st.session_state.active_menu == "🛠️ 模型重训":
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

        st.subheader("🛠️ 模型重训")
        st.info("通过上传历史数据和自定义标签字典，系统将重训并动态适配新的法定领域。")
        
        # ✅ 提供标准模板下载
        template_path = Path(project_root) / "mnx" / "classification" / "template_category_map.csv"
        if template_path.exists():
            with open(template_path, "rb") as f:
                st.download_button(
                    label="⬇️ 下载标准标签映射表模板（推荐）",
                    data=f.read(),
                    file_name="标签映射表_标准模板.csv",
                    mime="text/csv",
                    help="下载使用此模板可确保包含所有17个类别（0-16）"
                )
        
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
                
                # ✅ 显示识别到的标签映射表供用户检查
                st.markdown("---")
                st.subheader("📋 识别到的标签映射表")
                
                # 构建映射字典
                mapping_dict = dict(zip(df_m[m_id_col], df_m[m_name_col]))
                
                # 显示统计信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总类别数", len(mapping_dict))
                with col2:
                    min_id = min(mapping_dict.keys()) if mapping_dict else 0
                    max_id = max(mapping_dict.keys()) if mapping_dict else 0
                    st.metric("ID范围", f"{min_id} - {max_id}")
                with col3:
                    st.metric("完整性", "✅ 已加载")
                
                # 显示完整的映射表
                mapping_df = pd.DataFrame(
                    list(mapping_dict.items()),
                    columns=["标签序号", "中文名称"]
                ).sort_values("标签序号").reset_index(drop=True)
                
                st.dataframe(
                    mapping_df,
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # ✅ 移除硬编码的类别检查，完全基于用户上传的标签文档
                st.info(f"💡 **系统将完全按照上传的标签文档进行训练和预测**\n\n"
                       f"• 共识别到 **{len(mapping_dict)}** 个类别\n"
                       f"• 类别ID范围: **{min_id}** 至 **{max_id}**\n"
                       f"• 模型将根据这些标签进行训练，预测时也只使用这些标签")
                
                st.markdown("---")

                if st.button("🚀 开始训练最新的 3层神经网络", type="primary"):
                    # ✅ 显示清理提示
                    st.info("🧹 **即将执行以下操作：**\n"
                           "1. 删除旧的模型文件 (mlp_model.pth, mlp_encoder.pkl)\n"
                           "2. 删除旧的类别映射文件 (category_map.json)\n"
                           "3. 使用新的标签映射表重新训练模型\n"
                           "4. 生成新的 category_map.json")
                    
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
    elif st.session_state.active_menu == "💰 拖欠工资分析":
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
    elif st.session_state.active_menu == "可视化显示":
        api_fangshanmap_visualization()
    elif st.session_state.active_menu == "多对一分析":
        api_multi2one()
    elif st.session_state.active_menu == "一对多分析":
        api_one2multi()

    elif st.session_state.active_menu == "多级预警机制":
        api_multilevel_warning()

def api_multi2one():
    """
    Tab: 多对一分析
    左侧标准点位按钮列表 + 右侧可编辑 data_editor 表格
    """
    st.session_state.active_tab = "多对一分析"
    st.subheader("🔍 多对一相似地址分析")

    # ---- 前置条件检查 ----
    raw_df = st.session_state.get("raw_df")
    if raw_df is None or raw_df.empty:
        st.warning("⚠️ 请先在【📂 数据导入与预测】标签页上传并完成数据处理。")
        return

    json_path = Path(project_root) / "mnx" / "extract" / "result.json"
    if not json_path.exists():
        st.error(f"❌ 未找到脱敏分析文件: {json_path.name}")
        st.info("请先执行【脱敏】步骤生成 NER 提取结果。")
        return

    # ---- 执行分析按钮 ----
    analysis_key = "multi2one_analyzed"

    if analysis_key not in st.session_state:
        st.info(f"""
        **当前待分析数据:** `{len(raw_df)}` 条 (已去重)
        
        📁 分析数据源: `result.json` (脱敏后的NER实体提取结果)
        
        - 基于 **difflib 字面相似度算法** 对地点/机构实体进行聚类
        - 自动识别同一地点的不同表述变体
        """)
        if st.button("🚀 开始多对一分析", type="primary", key="btn_multi2one"):
            with st.spinner("正在进行相似度聚类分析，请稍候..."):
                try:
                    row_to_loc_map, cluster_summary = multi2one(str(json_path))
                    
                    st.session_state.multi2one_row_map = row_to_loc_map
                    st.session_state.multi2one_summary = cluster_summary
                    st.session_state[analysis_key] = True
                    
                    total_clusters = len(cluster_summary)
                    total_mapped = len(row_to_loc_map)
                    st.success(f"✅ 分析完成！共识别 **{total_clusters}** 个标准点位，覆盖 **{total_mapped}** 条数据。")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 分析过程出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    # ---- 分析结果展示（左右布局）----
    row_to_loc_map = st.session_state.get("multi2one_row_map", {})
    cluster_summary = st.session_state.get("multi2one_summary", [])

    if not row_to_loc_map:
        st.warning("分析结果为空，可能是数据中没有有效的地点/机构实体。")
        return

    # === 构建带"相似地名"列的完整 DF ===
    full_df = raw_df.copy()
    full_df["相似地名"] = full_df.index.astype(str).map(
        lambda idx: row_to_loc_map.get(f"l{int(idx) + 1}", row_to_loc_map.get(str(int(idx) + 1), ""))
    )

    # has_loc_count = (full_df["相似地名"] != "").sum()
    # no_loc_count = len(full_df) - has_loc_count

    # === 左右分栏 ===
    left_col, right_col = st.columns([1, 4])

    with left_col:
        st.markdown("**📍 选择标准点位**")

        # 当前选中的点位（默认为 None 表示显示全部）
        if "m2o_selected_loc" not in st.session_state:
            st.session_state.m2o_selected_loc = None

        # "全部" 按钮
        is_all = st.session_state.m2o_selected_loc is None
        btn_type = "primary" if is_all else "secondary"
        if st.button(f"📋 **全部** ({len(full_df)})", key="m2o_btn_all", use_container_width=True, type=btn_type):
            st.session_state.m2o_selected_loc = None
            st.rerun()

        st.markdown("---")

        visible_clusters = [cs for cs in cluster_summary if cs['row_count'] >= 2]
        # 各点位按钮
        for i, cs in enumerate(visible_clusters):
            loc_name = cs['standard_loc']
            count = cs['row_count']
            
            is_selected = (st.session_state.get("m2o_selected_loc") == loc_name)
            btn_type = "primary" if is_selected else "secondary"

            # ---- 根据行数决定优先级与颜色 ----
            if count < 5:
                priority_label = "低优先级"
                color = "#2563eb"   # 蓝色
            elif count <= 10:
                priority_label = "中优先级"
                color = "#d97706"   # 橙色
            else:
                priority_label = "高优先级"
                color = "#dc2626"   # 红色

            # 按钮：普通 Streamlit 原生按钮
            label = f"{loc_name} ({count})"
            if st.button(label, key=f"m2o_btn_{i}", use_container_width=True,
                         type=btn_type,
                         help=f"包含 {len(cs.get('variants', []))} 个变体 | {priority_label}"):
                st.session_state.m2o_selected_loc = loc_name
                st.rerun()

            # 彩色优先级标签（紧贴按钮下方）
            st.markdown(
                f'<div style="text-align:right; font-size:12px; font-weight:600; '
                f'color:{color}; margin-top:-12px; margin-bottom:8px; '
                f'padding-right:4px;">▸ {priority_label}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown("**📊 统计信息**")
        single_count = len([cs for cs in cluster_summary if cs['row_count'] == 1])
        multi_count = len(visible_clusters)
        st.info(
            f"多行相似点位: **{multi_count}** 个\n"
            f"单独点位(已折叠): **{single_count}** 个\n"
        )

    with right_col:
        # 根据左侧选择过滤数据
        selected_loc = st.session_state.get("m2o_selected_loc")
        
        if selected_loc is not None:
            display_df = full_df[full_df["相似地名"] == selected_loc].copy()
            display_df = display_df.reset_index(drop=True)
            st.caption(f"📍 当前显示: **{selected_loc}** — 共 **{len(display_df)}** 条记录")
        else:
            display_df = full_df.copy()
            display_df = display_df.reset_index(drop=True)
            st.caption(f"📋 当前显示: **全部数据** — 共 **{len(display_df)}** 条记录")

        if display_df.empty:
            st.info("该点位暂无匹配数据。")
        else:
            # === 构建 data_editor 列配置（参照台账查看） ===
            editor_cols = list(display_df.columns)
            
            # 确定列配置
            col_config = {}
            
            dtype_map = {
                "int64": ("number", st.column_config.NumberColumn),
                "int32": ("number", st.column_config.NumberColumn),
                "float64": ("number", st.column_config.NumberColumn),
                "float32": ("number", st.column_config.NumberColumn),
                "object": ("text",st.column_config.TextColumn),
                "string": ("text", st.column_config.TextColumn),
                "bool": ("checkbox", st.column_config.CheckboxColumn),
                "datetime64[ns]": ("datetime", st.column_config.DatetimeColumn),
            }

            for c in editor_cols:
                if c == col_config.get("_index", "").__class__.__name__:
                    continue
                    
                dtype_str = str(display_df[c].dtype)
                base_type, col_class = dtype_map.get(dtype_str, ("text", st.column_config.TextColumn))
                
                # 根据列名做宽度微调
                width = "medium"
                if "内容" in c or "content" in c.lower() or "描述" in c:
                    width = "large"
                elif "相似地名" == c or "编号" in c:
                    width = "small"
                    
                col_config[c] = col_class(c, width=width)

            st.markdown('💡 *提示：您可以直接在表格中编辑单元格内容。*')
            
            edited_df = st.data_editor(
                display_df,
                key="m2o_editor",
                width="stretch",
                height=max(400, min(700, len(display_df) * 37)),
                column_config=col_config,
                use_container_width=True,
                hide_index=True
            )

    # ---- 底部工具栏 ----
    st.markdown("---")
    tool_col1, tool_col2, tool_col3 = st.columns([2, 2, 4])

    with tool_col1:
        if st.button("🔄 重新分析", key="btn_multi2one_redo"):
            keys_to_clear = [k for k in st.session_state if k.startswith("multi2one") or k.startswith("m2o_")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

    with tool_col2:
        export_buf = io.BytesIO()
        with pd.ExcelWriter(export_buf, engine='xlsxwriter') as writer:
            full_df.to_excel(writer, index=False, sheet_name='多对一分析结果')
        st.download_button(
            label="📥 导出完整结果",
            data=export_buf.getvalue(),
            file_name="多对一相似地址分析.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )

def api_one2multi():
    """
    Tab: 一对多分析
    左侧人名按钮列表 + 右侧可编辑 data_editor 表格
    基于result.json中的人名(PER)实体进行相似度聚类，
    识别同一人的不同投诉记录。
    """
    st.session_state.active_tab = "一对多分析"
    st.subheader("👤 一对多人名相似分析")

    # ---- 前置条件检查 ----
    raw_df = st.session_state.get("raw_df")
    if raw_df is None or raw_df.empty:
        st.warning("⚠️ 请先在【📂 数据导入与预测】标签页上传并完成数据处理。")
        return

    json_path = Path(project_root) / "mnx" / "extract" / "result.json"
    if not json_path.exists():
        st.error(f"❌ 未找到脱敏分析文件: {json_path.name}")
        st.info("请先执行【脱敏】步骤生成 NER 提取结果。")
        return

    # ---- 执行分析按钮 ----
    analysis_key = "one2multi_analyzed"

    if analysis_key not in st.session_state:
        st.info(f"""
        **当前待分析数据:** `{len(raw_df)}` 条 (已去重)
        
        📁 分析数据源: `result.json` (脱敏后的NER实体提取结果)
        
        - 基于 **difflib 字面相似度算法** 对**人名(PER)**实体进行聚类
        - 自动识别同一人的不同表述（如"王先生"/"李先生"等变体）
        - 汇总同一人名的所有关联投诉记录
        """)
        if st.button("🚀 开始一对多分析", type="primary", key="btn_one2multi"):
            with st.spinner("正在进行人名相似度聚类分析，请稍候..."):
                try:
                    row_to_name_map, person_summary = one2multi(str(json_path))
                    
                    st.session_state.o2m_row_map = row_to_name_map
                    st.session_state.o2m_summary = person_summary
                    st.session_state[analysis_key] = True
                    
                    total_names = len(person_summary)
                    total_mapped = len(row_to_name_map)
                    st.success(f"✅ 分析完成！共识别 **{total_names}** 个标准姓名，覆盖 **{total_mapped}** 条数据。")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ 分析过程出错: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    # ---- 分析结果展示（左右布局）----
    row_to_name_map = st.session_state.get("o2m_row_map", {})
    person_summary = st.session_state.get("o2m_summary", [])

    if not row_to_name_map:
        st.warning("分析结果为空，可能是数据中没有有效的人名实体。")
        return

    # === 构建带"相似人名"列的完整 DF ===
    full_df = raw_df.copy()
    full_df["相似人名"] = full_df.index.astype(str).map(
        lambda idx: row_to_name_map.get(f"l{int(idx) + 1}", 
                                         row_to_name_map.get(str(int(idx) + 1), ""))
    )

    has_name_count = (full_df["相似人名"] != "").sum()

    # === 左右分栏 ===
    left_col, right_col = st.columns([1, 4])

    with left_col:
        st.markdown("**👤 选择标准姓名**")

        # 当前选中的人名（默认 None 表示显示全部）
        if "o2m_selected_name" not in st.session_state:
            st.session_state.o2m_selected_name = None

        # "全部" 按钮
        is_all = st.session_state.o2m_selected_name is None
        btn_type = "primary" if is_all else "secondary"
        if st.button(f"📋 **全部** ({len(full_df)})", key="o2m_btn_all", use_container_width=True, type=btn_type):
            st.session_state.o2m_selected_name = None
            st.rerun()

        st.markdown("---")

        # 各人名按钮 — 仅显示有多个关联行的人名 (row_count >= 2)
        visiblePersons = [ps for ps in person_summary if ps['row_count'] >= 2]

        for i, ps in enumerate(visiblePersons):
            name = ps['standard_name']
            count = ps['row_count']
            is_selected = st.session_state.o2m_selected_name == name
            btn_type = "primary" if is_selected else "secondary"

            label = f"{name} ({count})"
            if st.button(label, key=f"o2m_btn_{i}", use_container_width=True, type=btn_type,
                         help=f"包含 {len(ps['variants'])} 个变体"):
                st.session_state.o2m_selected_name = name
                st.rerun()

        st.markdown("---")
        st.markdown("**📊 统计信息**")
        single_count = len([ps for ps in person_summary if ps['row_count'] == 1])
        multi_count = len(visiblePersons)
        st.info(
            f"多人关联姓名: **{multi_count}** 个\n"
            f"单独姓名(已折叠): **{single_count}** 个\n"
            f"已匹配: **{has_name_count}** 条"
        )

    with right_col:
        # 根据左侧选择过滤数据
        selected_name = st.session_state.get("o2m_selected_name")
        
        if selected_name is not None:
            display_df = full_df[full_df["相似人名"] == selected_name].copy()
            display_df = display_df.reset_index(drop=True)
            st.caption(f"👤 当前显示: **{selected_name}** — 共 **{len(display_df)}** 条记录")
        else:
            display_df = full_df.copy()
            display_df = display_df.reset_index(drop=True)
            st.caption(f"📋 当前显示: **全部数据** — 共 **{len(display_df)}** 条记录")

        if display_df.empty:
            st.info("该人名暂无匹配数据。")
        else:
            # === 构建 data_editor 列配置（与多对一一致） ===
            editor_cols = list(display_df.columns)
            col_config = {}

            dtype_map = {
                "int64": ("number", st.column_config.NumberColumn),
                "int32": ("number", st.column_config.NumberColumn),
                "float64": ("number", st.column_config.NumberColumn),
                "float32": ("number", st.column_config.NumberColumn),
                "object": ("text", st.column_config.TextColumn),
                "string": ("text", st.column_config.TextColumn),
                "bool": ("checkbox", st.column_config.CheckboxColumn),
                "datetime64[ns]": ("datetime", st.column_config.DatetimeColumn),
            }

            for c in editor_cols:
                if c == col_config.get("_index", "").__class__.__name__:
                    continue
                    
                dtype_str = str(display_df[c].dtype)
                base_type, col_class = dtype_map.get(dtype_str, ("text", st.column_config.TextColumn))
                
                width = "medium"
                if "内容" in c or "content" in c.lower() or "描述" in c:
                    width = "large"
                elif "相似人名" == c or "编号" in c:
                    width = "small"
                    
                col_config[c] = col_class(c, width=width)

            st.markdown('💡 *提示：您可以直接在表格中编辑单元格内容。*')
            
            edited_df = st.data_editor(
                display_df,
                key="o2m_editor",
                width="stretch",
                height=max(400, min(700, len(display_df) * 37)),
                column_config=col_config,
                use_container_width=True,
                hide_index=True
            )

    # ---- 底部工具栏 ----
    st.markdown("---")
    tool_col1, tool_col2, tool_col3 = st.columns([2, 2, 4])

    with tool_col1:
        if st.button("🔄 重新分析", key="btn_one2multi_redo"):
            keys_to_clear = [k for k in st.session_state if k.startswith("o2m_") or k.startswith("one2multi")]
            for k in keys_to_clear:
                del st.session_state[k]
            st.rerun()

    with tool_col2:
        export_buf = io.BytesIO()
        with pd.ExcelWriter(export_buf, engine='xlsxwriter') as writer:
            full_df.to_excel(writer, index=False, sheet_name='一对多分析结果')
        st.download_button(
            label="📥 导出完整结果",
            data=export_buf.getvalue(),
            file_name="一对多人名相似分析.xlsx",
            mime="application/vnd.ms-excel",
            use_container_width=True
        )

def api_multilevel_warning():
    """
    Tab: 多级预警机制
    基于 Qwen-Embedding + 余弦相似度的多级预警处理 + 分级预警标注
    """
    st.session_state.active_tab = "多级预警机制"
    st.subheader("🔥 多级预警机制")
    
    # 检查是否有分析结果
    warning_result = st.session_state.get("warning_analysis_result")
    if warning_result is None:
        st.warning("⚠️ 请先在【📂 数据导入与预测】标签页执行【多级预警分析】。")
        return
    
    # ✅ 正确写法: 先尝试取快照, 取不到再 fallback
    if "warning_raw_df_snapshot" in st.session_state:
        raw_df = st.session_state["warning_raw_df_snapshot"]
    else:
        raw_df = st.session_state.get("raw_df")
        
    if raw_df is None or raw_df.empty:
        st.warning("⚠️ 未找到原始数据。")
        return
    
    # 获取用户选择的日期列名
    date_col = st.session_state.get("warning_date_col", raw_df.columns[0])
    
    # ---- 辅助函数: 计算单个组的分级预警等级 ----
    def compute_warning_level(group_indices):
        """
        根据组内时间间隔和记录条数, 返回分级预警标签
        
        规则:
          🔴 红色: 最大间隔 > 30天 OR 组内 >= 10 条
          🟡 黄色: 最大间隔 > 20天 OR 组内 5~9 条
          🔵 蓝色: 最大间隔 > 10天 OR 组内 3~4 条
          ⚪ 无:   其他情况不标警
        """
        group_size = len(group_indices)
        
        # 提取组内的日期值并尝试解析
        dates_raw = raw_df.iloc[group_indices][date_col].tolist()
        dates_parsed = []
        for d in dates_raw:
            try:
                dt = pd.to_datetime(d, errors='coerce')
                if pd.notna(dt):
                    dates_parsed.append(dt)
            except Exception:
                continue
        
        # 计算最大间隔 (天)
        max_interval_days = 0
        if len(dates_parsed) >= 2:
            dates_sorted = sorted(dates_parsed)
            max_interval_days = (dates_sorted[-1] - dates_sorted[0]).days
        
        # ---- 判定预警级别 (满足任一条件即触发该级别, 高优先级先判断) ----
        # 🔴 最高级
        if max_interval_days > 30 or group_size >= 10:
            return "🔴超30天未解决/有≥10条类似投诉"
        # 🟡 中高级
        elif max_interval_days > 20 or (5 <= group_size < 10):
            return "🟡超20天未解决/有5-9条类似投诉"
        # 🔵 中级
        elif max_interval_days > 10 or (3 <= group_size < 5):
            return "🔵超10天未解决/有3-4条类似投诉"
        else:
            return ""  # 不触发预警
    
    # ---- 构建显示用 DataFrame, 组间空行分隔, 新增"分级预警"列 ----
    display_dfs = []
    
    for i, group_indices in enumerate(warning_result):
        group_df = raw_df.iloc[group_indices].copy()
        group_size = len(group_indices)
        
        # 计算本组的分级预警等级
        level = compute_warning_level(group_indices)
        
        # 插入"预警组号"列
        group_df.insert(
            0, 
            '预警组号', 
            f'🔥 第 {i+1} 组 (共 {group_size} 条相似记录)'
        )
        
        # 插入"分级预警"列 (放在最右侧)
        group_df['分级预警'] = level
        
        display_dfs.append(group_df)
        
        # 插入空行 DataFrame 用于组间分隔
        empty_row = pd.DataFrame([[None] * len(group_df.columns)], columns=group_df.columns)
        display_dfs.append(empty_row)
    
    # 合并所有 DataFrame
    final_display_df = pd.concat(display_dfs, ignore_index=True)
    
    # # 显示统计信息
    # total_groups = len(warning_result)
    # total_rows = sum(len(g) for g in warning_result)
    
    # # 统计各级别数量
    # red_count = sum(1 for i, _ in enumerate(warning_result) if "🔴" in compute_warning_level(_))
    # yellow_count = ...  # 类似统计
    # blue_count = ...
    
    # st.info(
    #     f"📊 分析结果: 共 **{total_groups}** 个相似组, 涉及 **{total_rows}** 条数据\n\n"
    #     f"🔴红色预警: **X** 组 | 🟡黄色预警: **Y** 组 | 🔵蓝色预警: **Z** 组"
    # )
    
    # 用 streamlit 原生表格显示
    st.dataframe(final_display_df, use_container_width=True, height=500)
    
    # ---- 导出功能 ----
    col_export_csv, col_export_excel = st.columns(2)
    
    with col_export_csv:
        csv_buffer = io.StringIO()
        final_display_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 导出 CSV",
            data=csv_buffer.getvalue(),
            file_name=f"多级预警分析结果_{time.strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col_export_excel:
        excel_buffer = io.BytesIO()
        final_display_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        st.download_button(
            label="📥 导出 Excel",
            data=excel_buffer.getvalue(),
            file_name=f"多级预警分析结果_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

def run_warning_analysis(df, text_column, similarity_threshold=0.8):
    """
    执行多级预警分析: 计算 embedding -> 余弦相似度 -> Union-Find 聚类
    
    参数:
        df: 输入 DataFrame
        text_column: 要分析的文本列名
        similarity_threshold: 相似度阈值, 默认 0.8
        
    返回:
        List[List[int]]: 每个元素是一个相似组的行索引列表
    """
    # 1. 加载或获取缓存的 embedding 模型
    model = get_embedding_model()
    
    # 2. 准备文本数据
    texts = df[text_column].astype(str).tolist()
    
    # 3. 批量生成 embedding 向量
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    # 4. 计算余弦相似度矩阵 (N x N)
    sim_matrix = cosine_similarity(embeddings)
    
    # 5. Union-Find 聚类
    n = len(df)
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # 路径压缩
            x = parent[x]
        return x
    
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    
    # 遍历上三角矩阵, 合并相似的行对
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i][j] >= similarity_threshold:
                union(i, j)
    
    # 6. 收集聚类结果 (只保留有多于1行的组, 即真正相似的组)
    groups_dict = {}
    for idx in range(n):
        root = find(idx)
        if root not in groups_dict:
            groups_dict[root] = []
        groups_dict[root].append(idx)
    
    # 过滤掉只有一个元素的组 (不相似的独立行不显示)
    result = [sorted(indices) for indices in groups_dict.values() if len(indices) > 1]
    
    # 按组大小降序排列
    result.sort(key=len, reverse=True)
    
    return result


@st.cache_resource
def get_embedding_model():
    """缓存 Qwen-Embedding 模型实例, 避免重复加载"""
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir.parent / "mnx" / "core_model" / "Qwen3-Embedding-0.6B"
    return SentenceTransformer(str(model_path))


if __name__ == "__main__":
    show_app()