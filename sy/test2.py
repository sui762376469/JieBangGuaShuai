import streamlit as st
import pandas as pd

# 初始化状态
if 'is_scrolling' not in st.session_state:
    st.session_state.is_scrolling = False

def start_scroll():
    st.session_state.is_scrolling = True

def pause_scroll():
    st.session_state.is_scrolling = False

# 构造测试数据
rows = []
for i in range(1, 60):
    rows.append({"序号": i, "内容": f"实时滚动数据条目 {i}"})
df = pd.DataFrame(rows)

# 转成纯HTML表格
table_html = df.to_html(index=False, classes="scroll-table", escape=False)

# CSS 自动向上滚动 + 保留手动拖动 - 改进版本
scroll_css = f'''
<style>
.scroll-wrap {{
    height: 350px;
    overflow-y: auto; /* 改为auto，允许手动滚动 */
    border: 1px solid #ddd;
    position: relative;
}}
.scroll-table {{
    width: 100%;
    border-collapse: collapse;
}}
/* 整个表格进行滚动动画 */
.scroll-table-wrapper {{
    animation: scrollUp 40s linear infinite;
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
/* 当动画暂停时，显示真实滚动条 */
.scroll-wrap:has(.scroll-table-wrapper[style*="paused"]) {{
    overflow-y: auto;
}}
/* 表格样式 */
.scroll-table th, .scroll-table td {{
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}}
.scroll-table th {{
    background-color: #f2f2f2;
}}
</style>
'''

# 将表格包装在动画容器中
animated_table_html = f'''
<div class="scroll-table-wrapper" id="scrollTable">
    {table_html}
    {table_html}  <!-- 重复一次表格内容，使滚动更平滑 -->
</div>
'''

# 输出CSS
st.markdown(scroll_css, unsafe_allow_html=True)

# 添加控制按钮
col1, col2 = st.columns(2)
with col1:
    st.button("▶️ 开始滚动", on_click=start_scroll, key="play_btn")
with col2:
    st.button("⏸️ 暂停滚动", on_click=pause_scroll, key="pause_btn")

# 输出表格
st.markdown(f'<div class="scroll-wrap">{animated_table_html}</div>', unsafe_allow_html=True)