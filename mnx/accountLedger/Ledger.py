import streamlit as st
import pandas as pd
import io
import time
import sqlite3
import json
from pathlib import Path
import random
from datetime import datetime

BASE_DIR = Path(__file__).parent
DB_NAME = BASE_DIR / "complaints_ledger.db"

def api_account_ledger():
    """
    显示 Tab 1: 台账查看
    包含历史线索台账的展示、筛选、编辑、删除和导出功能
    """
    st.session_state.active_tab = "台账查看"
    # st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.subheader("📋 历史线索台账 (SQLite直连)")

    # 添加手动刷新按钮
    refresh_col1, refresh_col2 = st.columns([1, 10])
    with refresh_col1:
        if st.button("🔄 刷新台账", key="manual_refresh_ledger"):
            st.session_state.db_refresh_trigger = True
            st.rerun()
    
    if st.session_state.db_refresh_trigger:
        st.session_state.db_data = load_from_db()
        st.session_state.db_refresh_trigger = False

    db_data = st.session_state.db_data.copy()

    # 初始化滚动状态
    if 'is_scrolling' not in st.session_state:
        st.session_state.is_scrolling = False
    if 'simulated_data' not in st.session_state:
        st.session_state.simulated_data = None
    if 'scroll_counter' not in st.session_state:
        st.session_state.scroll_counter = 0

    def start_scroll():
        """开始滚动并生成模拟数据"""
        st.session_state.is_scrolling = True
        st.session_state.scroll_counter = 0
        
        # 从现有数据中随机抽取30%作为模拟新数据
        if not db_data.empty:
            sample_size = max(1, int(len(db_data) * 0.3))
            sampled_data = db_data.sample(n=min(sample_size, len(db_data)), random_state=random.randint(0, 1000)).copy()
            
            # 为模拟数据生成新的ID和时间戳
            max_id = db_data['id'].max() if 'id' in db_data.columns else 0
            new_ids = list(range(max_id + 1, max_id + len(sampled_data) + 1))
            sampled_data['id'] = new_ids
            
            # 重置时间戳为当前时间
            sampled_data['create_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sampled_data['status'] = '已筛查'
            sampled_data['manual_review'] = '待复核'
            
            st.session_state.simulated_data = sampled_data
            st.success(f"✅ 已生成 {len(sampled_data)} 条模拟新数据")
        else:
            st.warning("⚠️ 没有现有数据可供模拟")

    def pause_scroll():
        """暂停滚动"""
        st.session_state.is_scrolling = False

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
            if st.button(" 保存台账修改", type="primary"):
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
        st.subheader(" 导出全量台账")
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            export_df = db_data.copy()
            export_df = export_df[['id', 'content', '预测类别', 'status', 'manual_review', 'create_time']]
            export_df.columns = ['数据编号', '投诉内容', '预测类别', '处理状态', '人工复核', '入库时间']
            export_df.to_excel(writer, index=False, sheet_name='筛查台账')
        st.download_button(label="一键导出全量分类台账 (Excel)", data=buffer.getvalue(),
                           file_name="12345涉检线索筛查台账.xlsx", mime="application/vnd.ms-excel")
    else:
        st.info("台账暂无数据，请从【数据导入与预测】标签页导入12345投诉文件。")
    
    # ========== 动态滚动表格功能 ==========
    st.markdown("---")
    st.subheader("📊 实时数据监控面板")
    st.info("💡 点击下方按钮开始实时监控新投诉数据的动态流入")
    
    # 控制按钮
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ 开始滚动", use_container_width=True, type="primary", key="start_scroll_btn"):
            start_scroll()
            st.rerun()
    with col2:
        if st.button("⏸️ 停止滚动", use_container_width=True, type="secondary", key="pause_scroll_btn"):
            pause_scroll()
            st.rerun()
    
    # 如果正在滚动，显示动态表格
    if st.session_state.is_scrolling and st.session_state.simulated_data is not None:
        # 构造测试数据 - 每隔随机n行插入新数据
        rows = []
        simulated_df = st.session_state.simulated_data.copy()
        
        # 为模拟数据添加映射后的类别名称
        if 'pred_category' in simulated_df.columns:
            simulated_df["预测类别"] = simulated_df["pred_category"].map(CATEGORY_MAP)
        
        for idx, row in simulated_df.iterrows():
            content_preview = str(row.get('content', ''))[:50] + "..." if len(str(row.get('content', ''))) > 50 else str(row.get('content', ''))
            category = row.get('预测类别', '未知')
            create_time = row.get('create_time', '')
            rows.append({
                "序号": row.get('id', idx),
                "投诉内容": content_preview,
                "预测类别": category,
                "入库时间": create_time,
                "状态": "新录入"
            })
        
        df_scroll = pd.DataFrame(rows)
        
        # 转成纯HTML表格
        table_html = df_scroll.to_html(index=False, classes="scroll-table", escape=False)
        
        # CSS 自动向上滚动
        scroll_css = f'''
        <style>
        .scroll-wrap {{
            height: 350px;
            overflow-y: auto;
            border: 1px solid rgba(0, 212, 255, 0.3);
            border-radius: 8px;
            position: relative;
            background: rgba(10, 14, 26, 0.6);
        }}
        .scroll-table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .scroll-table-wrapper {{
            animation: scrollUp 120s linear infinite;
            animation-play-state: {"running" if st.session_state.is_scrolling else "paused"};
            width: 100%;
        }}
        @keyframes scrollUp {{
            0% {{ 
                transform: translateY(0); 
            }}
            100% {{ 
                transform: translateY(-100%); 
            }}
        }}
        .scroll-table th, .scroll-table td {{
            border: 1px solid rgba(0, 212, 255, 0.2);
            padding: 8px;
            text-align: left;
            color: #e2e8f0;
        }}
        .scroll-table th {{
            background-color: rgba(0, 102, 255, 0.3);
            color: #00d4ff;
            font-weight: 600;
        }}
        .scroll-table tr:nth-child(even) {{
            background-color: rgba(20, 25, 45, 0.4);
        }}
        .scroll-table tr:hover {{
            background-color: rgba(0, 212, 255, 0.1);
        }}
        </style>
        '''
        
        # 将表格包装在动画容器中
        animated_table_html = f'''
        <div class="scroll-table-wrapper" id="scrollTable">
            {table_html}
            {table_html}
        </div>
        '''
        
        # 输出CSS和表格
        st.markdown(scroll_css, unsafe_allow_html=True)
        st.markdown(f'<div class="scroll-wrap">{animated_table_html}</div>', unsafe_allow_html=True)
        
        # 显示统计信息
        st.info(f"📈 当前监控中：共 **{len(df_scroll)}** 条新数据正在实时流入")
        
    elif not st.session_state.is_scrolling and st.session_state.simulated_data is not None:
        # 停止滚动后，显示静态的模拟数据表格
        st.success("⏸️ 滚动已停止，以下是模拟的新增数据预览")
        
        simulated_df = st.session_state.simulated_data.copy()
        if 'pred_category' in simulated_df.columns:
            simulated_df["预测类别"] = simulated_df["pred_category"].map(CATEGORY_MAP)
        
        display_cols = ['id', 'content', '预测类别', 'status', 'manual_review', 'create_time']
        available_cols = [col for col in display_cols if col in simulated_df.columns]
        
        st.dataframe(simulated_df[available_cols], use_container_width=True, hide_index=True)
        
        # 提示用户可以刷新台账来查看完整结果
        st.info("💡 点击上方「🔄 刷新台账」按钮，可将这些模拟数据合并到主台账中")
    
    # st.markdown('</div>', unsafe_allow_html=True)

def get_dynamic_category_map():
    """
    从 category_map.json 加载分类映射，如果失败则使用默认映射
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
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
# 数据库操作函数
# ==========================================
def init_db():
    """初始化数据库表"""
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
    """保存数据到数据库"""
    conn = sqlite3.connect(DB_NAME)
    db_df = df[[text_col_name, '预测类别', '处理状态', '人工复核']].copy()
    db_df.columns = ['content', 'pred_category', 'status', 'manual_review']

    cursor = conn.cursor()
    cursor.execute("DELETE FROM ledger")
    conn.commit()
    
    db_df.to_sql('ledger', conn, if_exists='append', index=False)
    conn.close()


def save_simulated_to_db(df):
    """保存模拟数据到数据库"""
    if df.empty:
        return
    
    conn = sqlite3.connect(DB_NAME)
    db_df = df[['content', 'pred_category', 'status', 'manual_review', 'create_time']].copy()
    
    # 追加到数据库
    db_df.to_sql('ledger', conn, if_exists='append', index=False)
    conn.close()


def load_from_db():
    """从数据库加载数据，如果有模拟数据则合并"""
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM ledger ORDER BY create_time DESC", conn)
    conn.close()
    
    # 如果有模拟数据且用户已停止滚动，则合并到主数据中
    if 'simulated_data' in st.session_state and st.session_state.simulated_data is not None:
        if not st.session_state.get('is_scrolling', False):
            # 将模拟数据合并到数据库
            simulated_df = st.session_state.simulated_data.copy()
            
            # 确保列一致
            required_cols = ['content', 'pred_category', 'status', 'manual_review', 'create_time']
            if all(col in simulated_df.columns for col in required_cols):
                # 添加到数据库
                save_simulated_to_db(simulated_df)
                # 清空模拟数据标记
                st.session_state.simulated_data = None
                st.session_state.is_scrolling = False
                # 重新从数据库加载（包含新增数据）
                conn = sqlite3.connect(DB_NAME)
                df = pd.read_sql_query("SELECT * FROM ledger ORDER BY create_time DESC", conn)
                conn.close()
                st.success("✅ 模拟数据已成功合并到台账中！")
    
    return df


def update_db_record(row_id, category_id, manual_review):
    """更新数据库记录"""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("UPDATE ledger SET pred_category = ?, manual_review = ? WHERE id = ?",
                   (category_id, manual_review, row_id))
    conn.commit()
    conn.close()


def delete_db_records(row_ids):
    """批量删除数据库记录"""
    if not row_ids: return
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    placeholders = ','.join(['?'] * len(row_ids))
    cursor.execute(f"DELETE FROM ledger WHERE id IN ({placeholders})", tuple(row_ids))
    conn.commit()
    conn.close()