import streamlit as st
from security import authenticate_user

def show_login():
    # 设置页面配置
    st.set_page_config(
        page_title="12345涉检线索智能筛查系统",
        page_icon="⚖️", 
        layout="centered"
    )
    
    # 极致放大版：宽度 680px，全量加大字号与间距
    st.markdown("""
    <style>
    /* 1. 居中大标题 - 字体再放大，增加字间距 */
    .login-title {
        text-align: center;
        color: #374151; 
        font-size: 42px; /* 字体加大到 42px */
        font-weight: bold;
        margin-bottom: 35px;
        letter-spacing: 3px; /* 增加字间距，更显气势 */
    }
    
    /* 2. Form 表单容器 - 加宽到与标题相近的尺寸 */
    [data-testid="stForm"] {
        max-width: 680px; /* 宽度加宽到 680px */
        margin: 0 auto;
        padding: 45px; /* 内边距相应变大 */
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(245, 158, 11, 0.2); 
    }
    
    /* 强制放大 Streamlit 原生输入框的 Label(标签) 和 Input(输入内容) */
    [data-testid="stForm"] label p {
        font-size: 20px !important; /* "用户名"、"密码" 字号加大到 20px */
        font-weight: 500 !important;
        color: #4b5563 !important;
    }
    [data-testid="stForm"] input {
        font-size: 18px !important; /* 输入框内的字号加大到 18px */
        padding: 14px !important; /* 输入框变厚 */
    }
    
    /* 3. 修饰 Form 里面的提交按钮 - 大号按钮 */
    [data-testid="stFormSubmitButton"] button {
        background: linear-gradient(135deg, #d97706, #fbbf24) !important; 
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 16px !important; /* 按钮变厚 */
        font-size: 22px !important; /* 登录字号加大到 22px */
        font-weight: bold !important;
        letter-spacing: 10px !important; /* "登 录" 两字进一步拉开 */
        transition: all 0.3s ease !important;
        width: 100%;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1); 
        margin-top: 25px !important; /* 增加按钮上方的呼吸空间 */
    }
    
    /* 按钮悬浮效果 */
    [data-testid="stFormSubmitButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4) !important; 
    }
    
    /* 4. 测试账号提示框 - 完美对齐上方宽度 */
    .test-account {
        max-width: 680px; /* 同步加宽到 680px */
        margin: 25px auto 0;
        padding: 20px;
        background: rgba(254, 252, 232, 0.8); 
        border-radius: 10px;
        text-align: center;
        font-size: 18px; /* 字号加大到 18px */
        color: #4b5563;
    }
    .test-account strong {
        color: #d97706; 
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 渲染标题
    st.markdown('<h1 class="login-title">⚖️ 12345 涉检线索智能筛查系统</h1>', unsafe_allow_html=True)
    
    # 直接写原生表单
    with st.form("login_form", clear_on_submit=True):
        username = st.text_input(
            "用户名",
            placeholder="请输入用户名",
            key="login_username",
            help="请输入您的用户名"
        )
        
        password = st.text_input(
            "密码",
            type="password",
            placeholder="请输入密码",
            key="login_password",
            help="请输入您的密码"
        )
        
        submit = st.form_submit_button(
            "登录",
            use_container_width=True,
            type="primary"
        )
        
        # 登录逻辑
        if submit:
            if username and password:
                success, error = authenticate_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.session_state.user_type = "user"
                    st.session_state.page = "app"
                    qp = st.query_params
                    qp["page"] = "app"
                    qp["logged_in"] = "true"
                    st.rerun()
                    
                else:
                    st.error(error)
            else:
                st.warning("请输入用户名和密码")
    
    # 测试账号信息
    st.markdown('<div class="test-account"><strong>测试账号：</strong> admin / 123456</div>', unsafe_allow_html=True)

show_login()