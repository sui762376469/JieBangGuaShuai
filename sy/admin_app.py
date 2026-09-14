import streamlit as st
from pathlib import Path
import sys

# 添加项目路径
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def show_admin_app():
    """管理员主界面 - 左侧菜单 + 右侧内容"""
    st.set_page_config(
        page_title="管理员控制台",
        page_icon="🔐",
        layout="wide"
    )
    
    # 检查登录状态
    if not st.session_state.get("logged_in") or st.session_state.get("user_type") != "admin":
        st.session_state.page = "login"
        st.rerun()
    
    # 全局样式
    st.markdown("""
    <style>
    /* 全局背景 */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #1a1f3a 50%, #0f172a 100%);
        font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
        color: #e2e8f0;
    }
    
    /* 隐藏默认装饰 */
    #MainMenu {visibility: hidden !important;}
    .stDeployButton {display:none !important;}
    footer {visibility: hidden !important;}
    #stDecoration {display:none !important;}
    header {visibility: hidden !important;}
    .stApp > header {display: none !important;}
    
    /* 顶部导航栏 */
    .admin-header {
        background: rgba(10, 14, 26, 0.9);
        backdrop-filter: blur(15px);
        padding: 1rem 2rem;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2rem;
    }
    
    .admin-title {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .admin-user-info {
        color: #cbd5e1;
        font-size: 1rem;
    }
    
    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 26, 0.95);
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    section[data-testid="stSidebar"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        margin-bottom: 0.5rem;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #0066ff 0%, #00ccff 100%);
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4);
        color: white;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: rgba(20, 25, 45, 0.8);
        color: #e2e8f0;
        border: 1px solid rgba(100, 100, 150, 0.3);
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        transform: translateX(5px);
    }
    
    /* 内容区域卡片 */
    .content-card {
        background: rgba(10, 14, 26, 0.6);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(0, 212, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.5rem;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #e2e8f0 !important;
    }
    
    p, span, div {
        color: #cbd5e1 !important;
    }
                
    </style>
    """, unsafe_allow_html=True)
    
    # 初始化 session state
    if "admin_active_menu" not in st.session_state:
        st.session_state.admin_active_menu = "📊 系统概览"
    
    # 顶部导航栏
    st.markdown(f'''
    <div class="admin-header">
        <div class="admin-title">🔐 管理员控制台</div>
        <div class="admin-user-info">
            👤 管理员: {st.session_state.get("user", "Admin")} | 
            <span style="color: #00d4ff;">● 在线</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # 左侧侧边栏
    with st.sidebar:
        st.markdown("### 📋 管理菜单")
        st.markdown("---")
        
        menu_options = [
            "📊 系统概览",
            "👥 用户管理",
            "📈 数据统计",
            "⚙️ 系统设置",
            "📝 操作日志",
            "🚪 退出登录"
        ]
        
        for option in menu_options:
            is_active = st.session_state.admin_active_menu == option
            btn_type = "primary" if is_active else "secondary"
            
            if st.button(option, key=f"admin_menu_{option}", use_container_width=True, type=btn_type):
                if option == "🚪 退出登录":
                    logout_admin()
                else:
                    st.session_state.admin_active_menu = option
                    st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #94a3b8; font-size: 0.9rem; margin-top: 2rem;">
            © 2025 12345涉检线索系统<br>
            管理员专用
        </div>
        """, unsafe_allow_html=True)
    
    # 右侧内容区域
    render_admin_content(st.session_state.admin_active_menu)

def logout_admin():
    """管理员退出登录"""
    st.session_state.logged_in = False
    st.session_state.user = None
    st.session_state.user_type = None
    st.session_state.page = "login"
    st.session_state.login_mode = "user"
    st.session_state.admin_active_menu = "📊 系统概览"
    st.rerun()

def render_admin_content(menu):
    """根据菜单选项渲染内容"""
    
    if menu == "📊 系统概览":
        render_system_overview()
    elif menu == "👥 用户管理":
        render_user_management()
    elif menu == "📈 数据统计":
        render_data_statistics()
    elif menu == "⚙️ 系统设置":
        render_system_settings()
    elif menu == "📝 操作日志":
        render_operation_logs()

def render_system_overview():
    """系统概览页面"""
    st.header("📊 系统运行概览")
    
    # 统计卡片
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #00d4ff; font-size: 2.5rem; margin: 0;">1,234</h3>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">总数据量</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #10b981; font-size: 2.5rem; margin: 0;">56</h3>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">今日新增</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #f59e0b; font-size: 2.5rem; margin: 0;">89%</h3>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">处理率</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="content-card" style="text-align: center;">
            <h3 style="color: #ef4444; font-size: 2.5rem; margin: 0;">12</h3>
            <p style="color: #94a3b8; margin: 0.5rem 0 0 0;">待复核</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 系统状态
    st.subheader("🖥️ 系统状态")
    col_status1, col_status2 = st.columns(2)
    
    with col_status1:
        st.success("✅ 数据库连接正常")
        st.success("✅ AI模型服务运行中")
    
    with col_status2:
        st.success("✅ GPU加速已启用")
        st.info("ℹ️ 系统版本: v1.0.0")

def render_user_management():
    """用户管理页面"""
    st.header("👥 用户管理")
    st.info("此功能模块正在开发中...")
    
    # 示例表格
    import pandas as pd
    sample_data = {
        "用户名": ["user1", "user2", "user3"],
        "角色": ["普通用户", "普通用户", "普通用户"],
        "最后登录": ["2025-05-06", "2025-05-05", "2025-05-04"],
        "状态": ["活跃", "活跃", "离线"]
    }
    df = pd.DataFrame(sample_data)
    st.dataframe(df, use_container_width=True)

def render_data_statistics():
    """数据统计页面"""
    st.header("📈 数据统计分析")
    st.info("此功能模块正在开发中...")

def render_system_settings():
    """系统设置页面"""
    st.header("⚙️ 系统设置")
    st.info("此功能模块正在开发中...")

def render_operation_logs():
    """操作日志页面"""
    st.header("📝 操作日志")
    st.info("此功能模块正在开发中...")

if __name__ == "__main__":
    show_admin_app()