import streamlit as st
import pandas as pd
import numpy as np

# ============================================================
# 1. 页面基础配置
# ============================================================
st.set_page_config(
    page_title="我的第一个 Streamlit 应用",
    page_icon="🌟",
    layout="wide",
)

# ============================================================
# 2. 标题与 Markdown 介绍
# ============================================================
st.title("🌟 欢迎来到 Streamlit 世界！")
st.write(
    """
    这是一个 **Streamlit 入门示例**，展示了常用组件的效果。

    Streamlit 让你只需要几行 Python 代码就能构建漂亮的交互式 Web 应用。
    """
)
st.divider()  # 分割线

# ============================================================
# 3. 侧边栏 (Sidebar)
# ============================================================
with st.sidebar:
    st.header("⚙️ 控制面板")
    st.write("在这里调整参数，右侧内容会实时更新。")

    # 主题颜色选择
    theme = st.selectbox("选择主题色", ["蓝色", "绿色", "橙色", "紫色"])
    color_map = {"蓝色": "#1f77b4", "绿色": "#2ca02c", "橙色": "#ff7f0e", "紫色": "#9467bd"}
    selected_color = color_map[theme]

    # 数据条数
    n_rows = st.slider("数据行数", min_value=5, max_value=100, value=20)

    # 是否显示图表
    show_chart = st.toggle("显示图表", value=True)

    # 显示一段小信息
    st.caption("💡 提示：调整左侧参数试试看！")

# ============================================================
# 4. 交互式控件演示
# ============================================================
st.subheader("📝 交互式控件演示")

col1, col2, col3 = st.columns(3)

with col1:
    name = st.text_input("你的名字", placeholder="请输入名字...")
    if name:
        st.success(f"你好，{name}！欢迎体验 Streamlit 🎉")

with col2:
    age = st.number_input("你的年龄", min_value=0, max_value=150, value=25)
    st.info(f"你的年龄是 {age} 岁")

with col3:
    favorite_lang = st.radio("你喜欢的编程语言", ["Python", "JavaScript", "Java", "Go"])
    st.write(f"👍 你选择了 **{favorite_lang}**")

st.divider()

# ============================================================
# 5. 生成示例数据并展示 DataFrame
# ============================================================
st.subheader("📊 数据展示")

np.random.seed(42)
df = pd.DataFrame({
    "姓名": [f"用户{i}" for i in range(1, n_rows + 1)],
    "语文": np.random.randint(60, 100, n_rows),
    "数学": np.random.randint(60, 100, n_rows),
    "英语": np.random.randint(60, 100, n_rows),
})
df["总分"] = df["语文"] + df["数学"] + df["英语"]
df["平均分"] = df["总分"] / 3

# 数据编辑器
edited_df = st.data_editor(
    df,
    use_container_width=True,
    height=300,
    num_rows="fixed",
    key="data_editor",
)

st.caption(f"共 {len(edited_df)} 条数据，你可以在表格中直接编辑数据！")

# 数据统计
st.write("### 数据统计")
st.dataframe(edited_df.describe(), use_container_width=True)

st.divider()

# ============================================================
# 6. 图表可视化
# ============================================================
if show_chart:
    st.subheader("📈 图表可视化")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.write("##### 各科平均分（柱状图）")
        avg_scores = edited_df[["语文", "数学", "英语"]].mean()
        st.bar_chart(avg_scores, color=selected_color)

    with chart_col2:
        st.write("##### 总分分布（折线图）")
        st.line_chart(edited_df["总分"].values, color=selected_color)

    # 饼图用 matplotlib
    st.write("##### 各科分数占比（饼图）")
    total_by_subject = edited_df[["语文", "数学", "英语"]].sum()
    st.bar_chart(total_by_subject, color=selected_color)

st.divider()

# ============================================================
# 7. 展开面板与进度条
# ============================================================
st.subheader("📦 更多功能")

with st.expander("点击展开：关于 Streamlit"):
    st.write("""
    Streamlit 是一个开源的 Python 框架，专为机器学习和数据科学设计。

    **核心特点：**
    - 🚀 快速原型开发
    - 🎨 内置美观 UI 组件
    - 📡 自动实时刷新
    - 🧩 丰富的图表和可视化支持

    **官网：** [https://streamlit.io](https://streamlit.io)
    """)
    st.code("pip install streamlit", language="bash")

with st.expander("点击展开：常用组件速查"):
    st.write("""
    | 组件 | 用途 |
    |------|------|
    | `st.title()` | 大标题 |
    | `st.write()` | 文本/Markdown |
    | `st.dataframe()` | 展示 DataFrame |
    | `st.line_chart()` | 折线图 |
    | `st.bar_chart()` | 柱状图 |
    | `st.sidebar` | 侧边栏 |
    | `st.columns()` | 分栏布局 |
    | `st.button()` | 按钮 |
    | `st.slider()` | 滑块 |
    | `st.selectbox()` | 下拉选择 |
    | `st.progress()` | 进度条 |
    """)

# 进度条动画
st.write("##### 进度条演示")
if st.button("▶️ 点击开始进度条", key="progress_btn"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i in range(100):
        progress_bar.progress(i + 1)
        status_text.text(f"加载中... {i + 1}%")
    status_text.text("✅ 加载完成！")

st.divider()

# ============================================================
# 8. 页脚
# ============================================================
st.caption("用 ❤️ 和 Streamlit 构建 | 祝你学习愉快！")
