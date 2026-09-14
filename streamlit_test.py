import streamlit as st
import pandas as pd
import numpy as np
# Definitive CSS selectors for Streamlit 1.45.1+
st.markdown("""
<style>
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    div[data-testid="stDecoration"] {
        display: none !important;
    }
    div[data-testid="stStatusWidget"] {
        visibility: hidden !important;
    }
</style>
""", unsafe_allow_html=True)
# ----------------------
# 模拟数据（替换为你的 SQLite 数据即可）
# ----------------------
def get_sample_data():
    texts = [
        "我家小区信号特别差，经常没网，打电话断线，网速特别慢，而且每天晚上高峰期更严重",
        "网络频繁断开，每天晚上7点准时掉线，影响工作学习",
        "营业厅服务态度不好",
        "网速非常慢，看视频卡顿，下载东西超慢",
        "投诉信号弱，房间里完全没信号，出门才有，电梯里更是完全无服务",
        "乱扣费问题，每月多扣话费，查询账单不清楚，联系客服也没用",
        "宽带安装超时，师傅迟迟不来，效率很低，已经等了三天",
        "断网问题严重，一天掉线十几次，无法正常使用网络",
        "客服电话打不进去",
        "网络延迟高，玩游戏卡顿严重",
        "信号覆盖差，电梯、地下室完全无服务",
        "话费套餐不清晰，被误导办理高价套餐",
        "网络不稳定，时好时坏，影响居家办公",
        "上门维修不及时，报修后无人处理",
        "网速不达标，办理500M实际只有100M",
    ]
    df = pd.DataFrame({"投诉原文": texts})
    return df

# ----------------------
# BM25 精简逻辑：超过 min_length 才处理
# ----------------------
def bm25_filter_text(text, min_length=50, keep_ratio=50):
    text = str(text).strip()
    # 字符数不足，不处理
    if len(text) < min_length:
        return text
    # 超过最小长度，执行精简
    words = text.split()
    keep_len = max(1, int(len(words) * keep_ratio / 100))
    return " ".join(words[:keep_len])

# ----------------------
# 脱敏逻辑
# ----------------------
def desensitize_text(text):
    s = str(text)
    s = s.replace("我家", "【隐私】")
    s = s.replace("小区", "【地址】")
    s = s.replace("营业厅", "【机构】")
    s = s.replace("客服", "【人员】")
    return s

# ----------------------
# 分页函数
# ----------------------
def paginate_df(df, page_size=10, page=1):
    total_pages = max(1, int(np.ceil(len(df) / page_size)))
    page = max(1, min(page, total_pages))
    start = (page - 1) * page_size
    end = start + page_size
    return df.iloc[start:end], total_pages, page

# ----------------------
# 页面初始化
# ----------------------
st.set_page_config(page_title="投诉分类工具", layout="wide")
st.title("📝 12345投诉线索预处理工具")

# 初始化 session
if "df_original" not in st.session_state:
    st.session_state.df_original = get_sample_data()
if "df_processed" not in st.session_state:
    st.session_state.df_processed = st.session_state.df_original.copy()

# ======================
# 左侧侧边栏
# ======================
with st.sidebar:
    st.header("⚙️ 预处理配置")

    # ========== BM25 配置 ==========
    st.subheader("BM25 文本精简")
    use_bm25 = st.checkbox("✅ 启用 BM25 过滤长文本（提升效率）", value=False)

    bm25_min_chars = None
    bm25_threshold = None

    if use_bm25:
        # 新增：最小字符数限制
        bm25_min_chars = st.number_input(
            "文本超过多少字符才执行 BM25",
            min_value=10, max_value=500, value=50, step=1
        )
        # BM25 保留比例
        bm25_threshold = st.slider("BM25 保留比例（%）", 10, 100, 50)
        st.caption(f"仅处理 ≥ {bm25_min_chars} 字符的文本，保留 {bm25_threshold}%")

    st.divider()

    # ========== 脱敏配置 ==========
    st.subheader("数据脱敏")
    use_desensitize = st.checkbox("✅ 启用敏感信息脱敏", value=False)

    st.divider()

    # ========== 执行预处理 ==========
    if st.button("🚀 执行 BM25 + 脱敏预处理"):
        with st.spinner("正在处理数据..."):
            df = st.session_state.df_original.copy()

            # BM25 处理
            if use_bm25:
                df["投诉原文"] = df["投诉原文"].apply(
                    lambda x: bm25_filter_text(x, bm25_min_chars, bm25_threshold)
                )
                st.success("✅ BM25 处理完成")

            # 脱敏处理
            if use_desensitize:
                df["投诉原文"] = df["投诉原文"].apply(desensitize_text)
                st.success("✅ 脱敏完成")

            st.session_state.df_processed = df

    st.divider()


# ======================
# 右侧主界面：标签页 + 分页表格
# ======================
tab1, tab2 = st.tabs(["📋 预处理后数据", "原始数据"])

# 预处理结果
with tab1:
    st.subheader("预处理结果（分页展示）")
    df_show = st.session_state.df_processed

    # 页码控制
    col1, col2 = st.columns(2)
    with col1:
        page = st.number_input("页码", min_value=1, value=1)
    with col2:
        page_size = st.selectbox("每页条数", [10, 20, 50], index=0)

    df_paged, total_pages, current_page = paginate_df(df_show, page_size, page)
    st.caption(f"第 {current_page} / {total_pages} 页｜共 {len(df_show)} 条")

    # 翻页按钮
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("⏮️ 第一页"):
            page = 1
    with c2:
        if st.button("◀️ 上一页") and current_page > 1:
            page -= 1
    with c3:
        if st.button("▶️ 下一页") and current_page < total_pages:
            page += 1
    with c4:
        if st.button("⏭️ 最后一页"):
            page = total_pages

    st.dataframe(df_paged, use_container_width=True, height=400)

# 原始数据
with tab2:
    st.subheader("原始数据")
    st.dataframe(st.session_state.df_original, use_container_width=True)