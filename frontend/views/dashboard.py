import streamlit as st
from datetime import datetime
from frontend.utils import get_dashboard_metrics

def show_dashboard():
    st.markdown("<h2 style='font-weight: 700; margin-bottom: 20px;'>Enterprise Intelligence Dashboard</h2>", unsafe_allow_html=True)
    
    # Load metrics from API
    with st.spinner("Loading metrics..."):
        metrics = get_dashboard_metrics()
        
    if not metrics:
        st.error("Failed to load dashboard metrics from backend.")
        return
        
    # Styling for metrics cards
    st.markdown("""
    <style>
        .metric-card {
            background: linear-gradient(135deg, #1f1f2e 0%, #151522 100%);
            padding: 24px;
            border-radius: 14px;
            border: 1px solid #33334d;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            text-align: center;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #5d3fd3;
        }
        .metric-title {
            color: #a0a0c0;
            margin: 0;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
        }
        .metric-value {
            color: #ffffff;
            margin: 12px 0 0 0;
            font-size: 36px;
            font-weight: 700;
            font-family: 'Outfit', 'Inter', sans-serif;
            background: linear-gradient(90deg, #ffffff 0%, #dcdcef 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .purple-accent {
            background: linear-gradient(90deg, #a78bfa 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .blue-accent {
            background: linear-gradient(90deg, #60a5fa 0%, #3b82f6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .green-accent {
            background: linear-gradient(90deg, #34d399 0%, #059669 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .orange-accent {
            background: linear-gradient(90deg, #f59e0b 0%, #d97706 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
    </style>
    """, unsafe_allow_html=True)

    # 4 Columns for KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">Indexed PDFs</h3>
            <p class="metric-value purple-accent">{metrics['total_pdfs']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">Semantic Chunks</h3>
            <p class="metric-value blue-accent">{metrics['total_chunks']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">Conversations</h3>
            <p class="metric-value green-accent">{metrics['total_conversations']}</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 class="metric-title">Storage Used</h3>
            <p class="metric-value orange-accent">{metrics['storage_used_mb']} MB</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    # Bottom Layout: Split column for Recent Docs and Recent Questions
    left_col, right_col = st.columns(2)
    
    with left_col:
        st.markdown("<h4 style='color: #a78bfa; margin-bottom: 15px;'>Recent Documents</h4>", unsafe_allow_html=True)
        recent_docs = metrics.get("recent_documents", [])
        if not recent_docs:
            st.info("No documents uploaded yet. Go to the Documents manager to get started.")
        else:
            # Render a premium table for recent docs
            for doc in recent_docs:
                try:
                    dt = datetime.fromisoformat(doc["upload_time"].replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%b %d, %Y - %I:%M %p")
                except Exception:
                    formatted_time = doc["upload_time"]
                    
                st.markdown(f"""
                <div style="background-color: #1a1a26; padding: 14px 18px; border-radius: 8px; border-left: 3px solid #8b5cf6; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-weight: 500; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%;">{doc['filename']}</div>
                    <div style="color: #7c7c9c; font-size: 12px;">{formatted_time}</div>
                </div>
                """, unsafe_allow_html=True)
                
    with right_col:
        st.markdown("<h4 style='color: #60a5fa; margin-bottom: 15px;'>Recent Queries</h4>", unsafe_allow_html=True)
        recent_queries = metrics.get("recent_questions", [])
        if not recent_queries:
            st.info("No chats recorded yet. Start a new conversation on the Chat screen.")
        else:
            for q in recent_queries:
                try:
                    dt = datetime.fromisoformat(q["created_at"].replace("Z", "+00:00"))
                    formatted_time = dt.strftime("%b %d, %I:%M %p")
                except Exception:
                    formatted_time = q["created_at"]
                    
                st.markdown(f"""
                <div style="background-color: #1a1a26; padding: 14px 18px; border-radius: 8px; border-left: 3px solid #3b82f6; margin-bottom: 10px;">
                    <div style="font-weight: 500; font-size: 14px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 4px;">"{q['question']}"</div>
                    <div style="color: #7c7c9c; font-size: 11px; text-align: right;">{formatted_time}</div>
                </div>
                """, unsafe_allow_html=True)
