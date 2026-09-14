import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px

# ===== 配置区 =====
DB_PATH = r"./news_data1.db"
TABLE_NAME = "news_data"
PAGE_SIZE = 20  # 每页显示条数
# ==================


@st.cache_resource
def get_connection():
    """获取数据库连接（缓存复用）"""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def query_labels(conn):
    """获取所有标签及对应数量"""
    df = pd.read_sql_query(
        f"SELECT label, COUNT(*) as count FROM {TABLE_NAME} GROUP BY label ORDER BY count DESC",
        conn
    )
    return df


def query_data(conn, label_filter=None, keyword="", offset=0, limit=PAGE_SIZE):
    """查询数据，支持按标签筛选和关键词搜索"""
    sql = f"SELECT id, label, text FROM {TABLE_NAME} WHERE 1=1"
    params = []

    if label_filter and label_filter != "全部":
        sql += " AND label = ?"
        params.append(label_filter)

    if keyword.strip():
        sql += " AND text LIKE ?"
        params.append(f"%{keyword.strip()}%")

    # 获取总数
    count_sql = sql.replace("SELECT id, label, text", "SELECT COUNT(*)")
    total = pd.read_sql_query(count_sql, conn, params=params).iloc[0, 0]

    # 分页查询
    sql += " ORDER BY id LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    df = pd.read_sql_query(sql, conn, params=params)
    return df, total


def main():
    st.set_page_config(page_title="新闻数据浏览", layout="wide")
    conn = get_connection()

    # ---- 页面标题 ----
    st.title("📰 新闻数据浏览器")

    # ---- 统计概览 ----
    label_stats = query_labels(conn)
    total_count = label_stats["count"].sum()
    col1, col2, col3 = st.columns(3)
    col1.metric("总记录数", f"{total_count:,}")
    col2.metric("标签类别数", len(label_stats))
    col3.metric("数据库表", TABLE_NAME)

    # ---- 标签分布图 ----
    with st.expander("📊 标签分布", expanded=False):
        fig = px.bar(
            label_stats, x="label", y="count",
            title="各类别新闻数量分布",
            color="label",
            text_auto=True,
        )
        fig.update_layout(xaxis_title="标签", yaxis_title="数量")
        st.plotly_chart(fig, use_container_width=True)

        # 饼图
        fig_pie = px.pie(
            label_stats, values="count", names="label",
            title="标签占比",
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ---- 筛选区域 ----
    all_labels = ["全部"] + label_stats["label"].tolist()

    col_filter1, col_filter2 = st.columns([1, 3])
    with col_filter1:
        selected_label = st.selectbox("按标签筛选", all_labels)
    with col_filter2:
        search_keyword = st.text_input("🔍 搜索文本内容（模糊匹配）", placeholder="输入关键词...")

    # ---- 分页 ----
    df, total = query_data(conn, selected_label, search_keyword, 0, PAGE_SIZE)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)

    col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
    with col_page1:
        if st.button("⬅ 上一页", disabled=(st.session_state.get("page", 1) <= 1)):
            st.session_state["page"] = st.session_state.get("page", 1) - 1
            st.rerun()
    with col_page2:
        current_page = st.session_state.get("page", 1)
        st.markdown(
            f"<div style='text-align:center; padding: 8px;'>"
            f"第 **{current_page}** / **{total_pages}** 页 · 共 **{total}** 条记录"
            f"</div>",
            unsafe_allow_html=True,
        )
    with col_page3:
        if st.button("下一页 ➡", disabled=(current_page >= total_pages)):
            st.session_state["page"] = st.session_state.get("page", 1) + 1
            st.rerun()

    # ---- 数据表格 ----
    current_page = st.session_state.get("page", 1)
    offset = (current_page - 1) * PAGE_SIZE
    df, _ = query_data(conn, selected_label, search_keyword, offset, PAGE_SIZE)

    if df.empty:
        st.info("没有找到匹配的记录")
    else:
        # 重命名列显示
        df_display = df.rename(columns={"id": "ID", "label": "标签", "text": "文本内容"})
        st.dataframe(
            df_display,
            use_container_width=True,
            height=600,
            column_config={
                "ID": st.column_config.NumberColumn("ID", width="small"),
                "标签": st.column_config.TextColumn("标签", width="medium"),
                "文本内容": st.column_config.TextColumn("文本内容", width="large"),
            },
        )

        # ---- 点击查看详情 ----
        st.divider()
        st.subheader("📝 查看详情")
        selected_id = st.selectbox(
            "选择记录 ID",
            options=df["id"].tolist(),
            format_func=lambda x: f"ID: {x}  |  标签: {df[df['id']==x]['label'].values[0]}  |  {df[df['id']==x]['text'].values[0][:60]}...",
        )
        if selected_id:
            detail = df[df["id"] == selected_id].iloc[0]
            st.info(f"**标签：** {detail['label']}\n\n**文本内容：**\n\n{detail['text']}")


if __name__ == "__main__":
    main()
