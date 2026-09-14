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

# ==========================================
# 模块0：全局配置与动态映射字典
# ==========================================
DB_NAME = "complaints_ledger.db"

def get_dynamic_category_map():
    map_path = project_root / 'mnx' / 'classification' / 'category_map.json'
    default_map = {
        0: "非公益诉讼", 1: "生态环境和资源保护", 2: "食品药品安全", 3: "国有财产保护",
        4: "国有土地使用权出让", 5: "英雄烈士保护", 6: "未成年人保护", 7: "安全生产",
        8: "个人信息保护", 9: "军人地位和权益保障", 10: "反垄断和反不正当竞争",
        11: "反电信网络诈骗", 12: "农产品质量安全", 13: "妇女权益保障",
        14: "无障碍环境建设", 15: "文物和文化遗产保护", 16: "国防和军事利益保护",
        17: "其他领域一", 18: "其他领域二", 19: "拖欠工资"
    }

    if map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                loaded_map = json.load(f)
            return {int(k): str(v) for k, v in loaded_map.items()}
        except Exception:
            return default_map
    return default_map


CATEGORY_MAP = get_dynamic_category_map()
REVERSE_CATEGORY_MAP = {v: k for k, v in CATEGORY_MAP.items()}


# ==========================================
# 模块0.5：拖欠工资关键信息提取算法（全面升级版）
# ==========================================
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
# 模块1：数据库持久化交互逻辑
# ==========================================
def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
                   CREATE TABLE IF NOT EXISTS ledger
                   (
                       id INTEGER PRIMARY KEY AUTOINCREMENT,
                       content TEXT,
                       pred_category INTEGER,
                       status TEXT,
                       manual_review TEXT,
                       create_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                   )
                   ''')
    conn.commit()
    conn.close()


def save_to_db(df, text_col_name):
    conn = sqlite3.connect(DB_NAME)
    db_df = df[[text_col_name, '预测类别', '处理状态', '人工复核']].copy()
    db_df.columns = ['content', 'pred_category', 'status', 'manual_review']


    # db_df.to_sql('ledger', conn, if_exists='append', index=False)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM ledger")
    conn.commit()
    
    db_df.to_sql('ledger', conn, if_exists='append', index=False)
    conn.close()
    
    conn.close()


def load_from_db():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM ledger ORDER BY create_time DESC", conn)
    conn.close()
    return df


def update_db_record(row_id, category_id, manual_review):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE ledger SET pred_category = ?, manual_review = ? WHERE id = ?",
                   (category_id, manual_review, row_id))
    conn.commit()
    conn.close()


def delete_db_records(row_ids):
    if not row_ids: return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(row_ids))
    cursor.execute(f"DELETE FROM ledger WHERE id IN ({placeholders})", tuple(row_ids))
    conn.commit()
    conn.close()


# ==========================================
# 模块3：UI 与 交互流转
# ==========================================
def show_app():
    st.set_page_config(page_title="12345涉检线索智能筛查系统", layout="wide")
    init_db()

    st.markdown("""
        <style>
            .reportview-container { margin-top: -2em; }
            #MainMenu {visibility: hidden;}
            .stDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """, unsafe_allow_html=True)

    if "show_wage_panel" not in st.session_state:
        st.session_state.show_wage_panel = False
    if "db_refresh_trigger" not in st.session_state:
        st.session_state.db_refresh_trigger = True

    # ---------------- 顶部标题与触发按钮 ----------------
    title_col, btn_col = st.columns([8, 2])
    with title_col:
        st.title("⚖️ 12345 涉检线索智能筛查系统")
    with btn_col:
        st.write("") 
        st.write("")
        if st.button("💰 拖欠工资专项分析", type="secondary"):
            st.session_state.show_wage_panel = not st.session_state.show_wage_panel
            st.rerun()

    st.markdown("---")

    # ---------------- 左侧侧边栏 ----------------
    with st.sidebar:
        st.header("🖥️ 系统算力监测")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            st.success(f"**加速引擎开启** 🚀\n当前硬件: {gpu_name}")
        else:
            st.warning("**CPU 推理模式** 🐌\n未检测到可用独显。")

        st.markdown("---")
        st.header("⚙️ 数据操作流水线")

        with st.expander("🛠️ 进阶：AI 模型在线重训", expanded=False):
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

        st.markdown("---")

        uploaded_file = st.file_uploader("📂 导入12345原始待预测数据", type=["csv", "xlsx", "xls"])

        if uploaded_file is not None:
            if "raw_df" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                if uploaded_file.name.endswith('.csv'):
                    st.session_state.raw_df = pd.read_csv(uploaded_file)
                else:
                    st.session_state.raw_df = pd.read_excel(uploaded_file)
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.bm25_df = None
                st.session_state.desensitized_df = None

            pred_text_col = st.selectbox("🎯 指定待预测数据中的 '特征文本' 列", st.session_state.raw_df.columns, index=0)

            # st.subheader("✂️ 第一步：BM25 提取重点")
            # if st.button("📊 开始提取核心摘要", type="primary"):
            #     with st.spinner("正在调用 BM25 算法分析重要句子..."):
            #         try:
            #             df_res = api_bm25_df(st.session_state.raw_df, text_column=pred_text_col)
            #             st.session_state.bm25_df = df_res
            #             st.success("摘要提取完成！")
            #         except Exception as e:
            #             st.error(f"提取过程出错: {str(e)}")
            df_res = api_bm25_df(st.session_state.raw_df, text_column=pred_text_col)
            st.session_state.bm25_df = df_res


            if st.session_state.get("bm25_df") is not None:
                st.markdown("---")
                st.subheader("🛠️ 第一步：执行脱敏")
                if st.button("🛡️ 对摘要执行专业脱敏", type="primary"):
                    with st.spinner("正在清洗摘要数据..."):
                        try:
                            df_desen = api_desensitize_df(st.session_state.bm25_df, text_column=pred_text_col)
                            st.session_state.desensitized_df = df_desen
                            st.success("脱敏清洗完成！")
                        except Exception as e:
                            st.error(f"脱敏过程出错: {str(e)}")

                if st.session_state.get("desensitized_df") is not None:
                    st.markdown("---")
                    st.subheader("🧠 第二步：真实的AI大模型推理")
                    if st.button("🚀 调用 3层MLP 引擎预测并入库"):
                        with st.spinner("正在将文本投入大模型获取特征向量并进行推理，请耐心等待..."):
                            df_to_predict = st.session_state.desensitized_df.copy()
                            texts_to_predict = df_to_predict[pred_text_col].tolist()

                            try:
                                pred_labels_str = api_predict_categories(texts_to_predict)
                                predictions = []
                                for label_str in pred_labels_str:
                                    cat_id = REVERSE_CATEGORY_MAP.get(label_str, 0)
                                    predictions.append(cat_id)

                                df_to_predict["预测类别"] = predictions
                                df_to_predict["处理状态"] = "已筛查"
                                df_to_predict["人工复核"] = "待复核"

                                save_to_db(df_to_predict, pred_text_col)
                                st.session_state.db_refresh_trigger = True
                                st.success("🎉 AI 预测并落库成功！请在右侧查看台账。")
                            except Exception as e:
                                st.error(f"模型推理失败: {e}")

    # ---------------- 主题区域布局：动态列实现"侧滑" ----------------
    if st.session_state.show_wage_panel:
        col_main, col_right = st.columns([6, 4]) 
    else:
        col_main = st.container()
        col_right = None

    # ====== [主界面区域] 历史台账 ======
    with col_main:
        st.subheader("📋 历史线索台账 (SQLite直连)")

        if st.session_state.db_refresh_trigger:
            st.session_state.db_data = load_from_db()
            st.session_state.db_refresh_trigger = False

        db_data = st.session_state.db_data.copy()

        if not db_data.empty:
            db_data["预测类别"] = db_data["pred_category"].map(CATEGORY_MAP)
            for col in ["content", "预测类别", "manual_review", "status", "create_time"]:
                db_data[col] = db_data[col].astype(str)

            inner_col1, inner_col2 = st.columns([2, 4])
            with inner_col1:
                category_options = ["全部"] + list(CATEGORY_MAP.values())
                category_filter = st.selectbox("按预测类别筛选", category_options)

            display_df = db_data.copy()
            if category_filter != "全部":
                display_df = display_df[display_df["预测类别"] == category_filter]

            display_df = display_df.set_index('id')
            display_df["删除"] = False

            st.markdown('💡 *提示：您可以双击单元格修改"人工复核"和"预测类别"，或勾选最前方的选框进行数据删除。*')

            editable_cols = ["删除", "content", "预测类别", "manual_review", "status", "create_time"]
            editor_key = f"ledger_editor_{category_filter}"

            edited_df = st.data_editor(
                display_df[editable_cols],
                key=editor_key,
                width="stretch",
                disabled=["content", "status", "create_time"],
                column_config={
                    "_index": st.column_config.NumberColumn("数据编号", width="small"),
                    "删除": st.column_config.CheckboxColumn("🗑️ 勾选删除", default=False, width="small"),
                    "content": st.column_config.TextColumn("投诉内容 (已精简脱敏)", width="large"),
                    "预测类别": st.column_config.SelectboxColumn(
                        "预测类别", options=list(CATEGORY_MAP.values()), required=True, width="medium"
                    ),
                    "manual_review": st.column_config.SelectboxColumn(
                        "人工复核", options=["待复核", "通过", "驳回"], required=True, width="small"
                    ),
                    "status": st.column_config.TextColumn("系统状态", width="small"),
                    "create_time": st.column_config.TextColumn("入库时间", width="medium")
                }
            )

            btn_col1, btn_col2, _ = st.columns([3, 3, 4])
            with btn_col1:
                if st.button("💾 保存台账修改", type="primary"):
                    with st.spinner("正在同步修改..."):
                        changes_made = 0
                        for row_id, row in edited_df.iterrows():
                            if row.get("删除", False):
                                continue
                            original_row = display_df.loc[row_id]
                            if (row['预测类别'] != original_row['预测类别']) or (row['manual_review'] != original_row['manual_review']):
                                new_category_id = REVERSE_CATEGORY_MAP.get(row['预测类别'])
                                if new_category_id is None: continue
                                update_db_record(row_id, new_category_id, row['manual_review'])
                                changes_made += 1
                        if changes_made > 0:
                            st.success(f"成功同步了 {changes_made} 条修改记录！")
                            st.session_state.db_refresh_trigger = True
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.info("未检测到修改。")

            with btn_col2:
                if st.button("🗑️ 批量删除选中数据", type="secondary"):
                    ids_to_delete = edited_df[edited_df["删除"] == True].index.tolist()
                    if ids_to_delete:
                        with st.spinner("正在物理删除数据..."):
                            delete_db_records(ids_to_delete)
                            st.success(f"成功删除了 {len(ids_to_delete)} 条记录！")
                            st.session_state.db_refresh_trigger = True
                            time.sleep(1)
                            st.rerun()
                    else:
                        st.warning("⚠️ 请先在表格最左侧勾选数据。")

            st.markdown("---")
            st.subheader("📥 导出全量台账")
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                export_df = db_data.copy()
                export_df = export_df[['id', 'content', '预测类别', 'status', 'manual_review', 'create_time']]
                export_df.columns = ['数据编号', '投诉内容', '预测类别', '处理状态', '人工复核', '入库时间']
                export_df.to_excel(writer, index=False, sheet_name='筛查台账')
            st.download_button(label="一键导出全量分类台账 (Excel)", data=buffer.getvalue(),
                               file_name="12345涉检线索筛查台账.xlsx", mime="application/vnd.ms-excel")
        else:
            st.info("台账暂无数据，请从左侧导入12345投诉文件。")

    # ====== [右侧面板区域] 拖欠工资专项分析 ======
    if col_right:
        with col_right:
            # 👇 核心改动点升级：强制兼容所有 Streamlit 版本标签名，增加 !important，增加微边框阴影
            st.markdown('''
                <style>
                /* 覆盖新老版本的 column 选择器，强制赋予灰底、圆角与边框 */
                div[data-testid="stColumn"]:has(.target-right-panel),
                div[data-testid="column"]:has(.target-right-panel) {
                    background-color: #f0f2f6 !important;
                    padding: 1.5rem !important;
                    border-radius: 0.8rem !important;
                    border: 1px solid #e1e4e8 !important;
                    box-shadow: -2px 0 8px rgba(0,0,0,0.02) !important;
                }
                /* 针对内部可能存在的透明度问题做修复 */
                div[data-testid="stColumn"]:has(.target-right-panel) > div {
                    background-color: transparent !important; 
                }
                </style>
                <div class="target-right-panel"></div>
            ''', unsafe_allow_html=True)

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
                 st.warning("请先在左侧完成数据导入与推理。")

if __name__ == "__main__":
    show_app()