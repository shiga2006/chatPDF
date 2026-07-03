import streamlit as st
from frontend.utils import login_user, register_user
from frontend.views.dashboard import show_dashboard
from frontend.views.documents import show_documents
from frontend.views.chat import show_chat
from frontend.views.settings import show_settings

# 1. Config page settings
st.set_page_config(
    page_title="Enterprise Agentic Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Setup Session State keys
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Dashboard"

# Load custom Google Font properties
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# 3. RENDER AUTH VIEW IF UNVERIFIED
if not st.session_state.token:
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.8, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h1 style="font-weight: 800; font-size: 32px; background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Enterprise Agentic AI
            </h1>
            <p style="color: #7c7c9c; font-size: 14px; margin-top: 4px;">
                Internal Knowledge Base Assistant (RAG & LangGraph)
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        tab_login, tab_register = st.tabs(["🔑 Access Account", "📝 Register New Account"])
        
        with tab_login:
            login_email = st.text_input("Corporate Email", key="login_email_input")
            login_password = st.text_input("Password", type="password", key="login_pass_input")
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Log In", use_container_width=True, type="primary"):
                if not login_email or not login_password:
                    st.error("Please enter email and password.")
                else:
                    with st.spinner("Authenticating credentials..."):
                        auth_data = login_user(login_email, login_password)
                    if auth_data:
                        st.session_state.token = auth_data["access_token"]
                        st.session_state.user = {
                            "name": auth_data["name"],
                            "email": auth_data["email"]
                        }
                        st.toast(f"Welcome back, {auth_data['name']}!")
                        st.rerun()
                        
        with tab_register:
            reg_name = st.text_input("Full Name", key="reg_name_input")
            reg_email = st.text_input("Corporate Email Address", key="reg_email_input")
            reg_password = st.text_input("Create Password", type="password", key="reg_pass_input")
            
            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            if st.button("Create Account", use_container_width=True, type="primary"):
                if not reg_name or not reg_email or not reg_password:
                    st.error("Please complete all input fields.")
                elif len(reg_password) < 6:
                    st.error("Password must be at least 6 characters.")
                else:
                    with st.spinner("Creating account credentials..."):
                        auth_data = register_user(reg_name, reg_email, reg_password)
                    if auth_data:
                        st.session_state.token = auth_data["access_token"]
                        st.session_state.user = {
                            "name": auth_data["name"],
                            "email": auth_data["email"]
                        }
                        st.success("Account created successfully!")
                        st.rerun()

# 4. RENDER WORKSPACE APPLICATION VIEW IF AUTHENTICATED
else:
    # --- SIDEBAR HEADER ---
    st.sidebar.markdown(f"""
    <div style="padding: 10px 0; margin-bottom: 20px;">
        <h3 style="margin:0; font-weight:700; background: linear-gradient(90deg, #a78bfa 0%, #60a5fa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Agentic Assistant</h3>
        <p style="margin: 2px 0 0 0; font-size:11px; color:#7c7c9c; text-transform: uppercase; letter-spacing: 1px;">Enterprise Knowledge Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- NAVIGATION BUTTONS ---
    nav_selection = st.sidebar.radio(
        "Navigation",
        options=["Dashboard", "Documents Manager", "Chat Workspace", "System Settings"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("<hr style='border-color: #33334d; margin: 20px 0;'/>", unsafe_allow_html=True)
    
    # Nav page dispatcher (will override st.session_state.current_page)
    st.session_state.current_page = nav_selection
    
    # --- ROUTE TO APPROPRIATE VIEWS ---
    if st.session_state.current_page == "Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "Documents Manager":
        show_documents()
    elif st.session_state.current_page == "Chat Workspace":
        show_chat()
    elif st.session_state.current_page == "System Settings":
        show_settings()
        
    # --- SIDEBAR PROFILE FOOTER ---
    st.sidebar.markdown("<div style='height: 60px;'></div>", unsafe_allow_html=True)
    st.sidebar.markdown(f"""
    <div style="background-color: #1a1a26; padding: 12px 16px; border-radius: 8px; border: 1px solid #33334d;">
        <div style="font-weight: 600; font-size: 13px; color:#ffffff;">{st.session_state.user['name']}</div>
        <div style="font-size: 11px; color:#7c7c9c; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{st.session_state.user['email']}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    if st.sidebar.button("Logout", use_container_width=True, type="secondary"):
        st.session_state.token = None
        st.session_state.user = None
        st.session_state.current_session_id = None
        st.toast("Logged out successfully.")
        st.rerun()
