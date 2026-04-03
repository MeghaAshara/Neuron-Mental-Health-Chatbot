"""
app.py  —  Neuron Mental Health Analytics Platform
===================================================
Tabs:
  Tab 1 — Chat Support      (chatbot.py)
  Tab 2 — Risk Assessment   (ml_model.py)
  Tab 3 — Data Insights EDA (eda.py)
  Tab 4 — ML Model Info     (ml_model.py)
Run:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import chatbot    
import ml_model   
import eda        

st.set_page_config(
    page_title="Neuron — Mental Health Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: #1a2744;
}

.stApp {
    background:
        radial-gradient(ellipse at 0% 0%,   #c8f0e8 0%,  transparent 45%),
        radial-gradient(ellipse at 100% 0%,  #d4c8f5 0%,  transparent 45%),
        radial-gradient(ellipse at 100% 100%,#fde8c8 0%,  transparent 45%),
        radial-gradient(ellipse at 0% 100%,  #c8e8f5 0%,  transparent 45%),
        linear-gradient(135deg, #e8f5f0 0%, #ede8fc 35%, #fdf0e0 70%, #e0f0fa 100%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #e8f8f2 0%, #eae4f8 50%, #fef3e8 100%);
    border-right: 2px solid #b8ddd0;
}
[data-testid="stSidebar"] * { color: #1a3a2a !important; }
[data-testid="stSidebar"] b { color: #2d5a3d !important; }

/* ── Headings ── */
h1, h2, h3, h4 {
    font-family: 'Playfair Display', serif !important;
    color: #1a2744 !important;
    letter-spacing: -0.01em;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: linear-gradient(135deg, #d6f0e8 0%, #c8e8f8 100%);
    border: 1.5px solid #8ecfbb;
    border-radius: 14px;
    padding: 14px 18px;
    box-shadow: 0 3px 14px rgba(30,120,90,0.10);
}
[data-testid="metric-container"] label {
    color: #1a5c42 !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0f2d20 !important;
    font-weight: 800;
    font-size: 1.5rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2a9d8f 0%, #38b89a 100%);
    color: #fff !important;
    border: none;
    border-radius: 10px;
    padding: 9px 22px;
    font-weight: 700;
    font-family: 'Nunito', sans-serif;
    transition: all .18s ease;
    box-shadow: 0 3px 14px rgba(42,157,143,0.30);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(42,157,143,0.45);
    background: linear-gradient(135deg, #21867a 0%, #2da888 100%);
}

/* ── Inputs ── */
.stTextInput input {
    background: #e8f8f4 !important;
    border: 1.5px solid #7ecab8 !important;
    border-radius: 10px !important;
    color: #0f2d20 !important;
    caret-color: #2a9d8f !important;
    font-size: 1rem !important;
    font-family: 'Nunito', sans-serif !important;
    box-shadow: 0 1px 6px rgba(42,157,143,0.08) !important;
}
.stTextInput input::placeholder { color: #5a9e8c !important; opacity: 1 !important; }
.stTextInput input:focus {
    background: #d6f5ee !important;
    border-color: #2a9d8f !important;
    box-shadow: 0 0 0 3px rgba(42,157,143,0.18) !important;
    color: #0f2d20 !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] { color: #2a9d8f; }
[data-testid="stSlider"] div[role="slider"] {
    background: #2a9d8f !important;
    border-color: #2a9d8f !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #e8f8f4 !important;
    border-color: #7ecab8 !important;
    color: #0f2d20 !important;
}

/* ── Tabs ── */
[data-baseweb="tab-list"] {
    background: linear-gradient(90deg, #fde8d0 0%, #e8d8f8 50%, #d0f0e8 100%);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1.5px solid #f0c898;
}
[data-baseweb="tab"] {
    color: #5a3a1a !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    font-family: 'Nunito', sans-serif !important;
}
[aria-selected="true"] {
    background: linear-gradient(135deg, #2a9d8f, #38b89a) !important;
    color: #ffffff !important;
    box-shadow: 0 3px 10px rgba(42,157,143,0.30) !important;
}

/* ── Chat bubbles ── */
.bubble-user {
    background: linear-gradient(135deg, #d8eef8, #cce4f5);
    border: 1.5px solid #7dbcd8;
    border-radius: 18px 18px 4px 18px;
    padding: 13px 18px;
    margin: 10px 0 10px auto;
    max-width: 80%;
    color: #0f2840;
    line-height: 1.7;
    font-size: 0.97rem;
    box-shadow: 0 2px 12px rgba(30,90,140,0.10);
}
.bubble-bot {
    background: linear-gradient(135deg, #e4f8f0, #d8f2e8);
    border: 1.5px solid #7ecab8;
    border-radius: 18px 18px 18px 4px;
    padding: 13px 18px;
    margin: 10px auto 10px 0;
    max-width: 85%;
    color: #0f2d20;
    line-height: 1.7;
    font-size: 0.97rem;
    box-shadow: 0 2px 12px rgba(30,120,90,0.10);
}
.bubble-label {
    font-size: .75rem;
    color: #2a6858;
    margin-bottom: 4px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* ── Cards ── */
.card {
    background: linear-gradient(135deg, #fff8f0 0%, #fef4e8 100%);
    border: 1.5px solid #f5c888;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 16px;
    box-shadow: 0 3px 16px rgba(200,120,30,0.08);
}

/* ── Risk badges ── */
.badge { display:inline-block; padding:6px 22px; border-radius:999px; font-weight:800; font-size:.9rem; }
.badge-Minimal  { background:#d4f7e8; color:#0a5c2e; border:2px solid #4ade80; }
.badge-Mild     { background:#fef6c8; color:#7a4a00; border:2px solid #f0b429; }
.badge-Moderate { background:#fde8d0; color:#8c3000; border:2px solid #f97316; }
.badge-Severe   { background:#fdd8d8; color:#8c0000; border:2px solid #ef4444; }

/* ── Objective checklist rows ── */
.obj-row {
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 10px 16px;
    margin: 6px 0;
    background: linear-gradient(90deg, #e8f8f2, #f0faf6);
}

p, li, span, label { color: #1a2744 !important; }

hr { border-color: #b8ddd0 !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: linear-gradient(135deg, #eef8f4, #e8f4fc);
    border: 1.5px solid #8ecfbb !important;
    border-radius: 12px;
}

/* ── Info / warning boxes ── */
[data-testid="stAlert"] {
    background: linear-gradient(135deg, #e0f4ec, #d8eef8) !important;
    border: 1.5px solid #7ecab8 !important;
    color: #0f2d20 !important;
    border-radius: 10px;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #d6f0e8; border-radius: 99px; }
::-webkit-scrollbar-thumb { background: #7ecab8; border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: #2a9d8f; }

/* ── General text visibility ── */
.stMarkdown, .stText { color: #1a2744 !important; }
[data-testid="stForm"] {
    background: linear-gradient(135deg, #eef8f4 0%, #f8f0fc 100%);
    border: 1.5px solid #8ecfbb;
    border-radius: 16px;
    padding: 8px;
}
</style>
""", unsafe_allow_html=True)

# CACHED LOADERS
@st.cache_data
def load_dataset():
    return eda.load_data()

@st.cache_resource
def load_ml_model():
    return ml_model.load()

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:24px 0 16px;'>
        <div style='font-size:3rem;line-height:1'>🧠</div>
        <div style='font-family:Playfair Display,serif;font-size:1.9rem;font-weight:700;
                    background:linear-gradient(135deg,#2a9d8f,#38b89a);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    letter-spacing:-.02em;margin-top:6px;'>NEURON</div>
        <div style='color:#1a5c42;font-size:.8rem;margin-top:4px;font-weight:700;'>
            Mental Health Analytics Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("**🔧 Ollama Model**")
    available_models = chatbot.get_models()
    selected_model = st.selectbox("Select model", available_models, label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style='font-size:.78rem;line-height:1.8;color:#2a4a3a;'>
    <b style='color:#c0392b;'>⚠️ Disclaimer</b><br>
    Educational purposes only. Not a substitute for professional medical advice.
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style='font-size:.78rem;line-height:1.9;color:#2a4a3a;'>
    <b style='color:#1a5c42;'>🆘 Crisis Helplines (India)</b><br>
    iCall &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;9152987821<br>
    Vandrevala &nbsp;1860-2662-345<br>
    SNEHI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;044-24640050
    </div>
    """, unsafe_allow_html=True)

# PAGE HEADER
st.markdown("""
<div style='padding:8px 0 28px;'>
    <h1 style='color:#1a2744;font-size:2.3rem;margin:0;font-family:Playfair Display,serif;'>
        <span style='background:linear-gradient(135deg,#2a9d8f,#e76f51);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>🧠 Neuron
        </span>
        <span style='color:#1a2744;'> Mental Health Platform</span>
    </h1>
    <p style='color:#2a6858;margin:6px 0 0;font-size:1rem;font-weight:600;'>Your compassionate wellness companion</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "💬  Chat Support",
    "📋  Risk Assessment",
    "📊  Data Insights",
    "🤖  ML Model Info",
])

# TAB 1 ── CHAT SUPPORT  (uses chatbot.py)
with tab1:
    st.markdown("### 💬 Chat with Neuron")
    st.markdown(
        "<p style='color:#1a5c42;margin-top:-8px;margin-bottom:20px;font-weight:600;'>"
        "A safe, judgment-free space to discuss your mental wellbeing.</p>",
        unsafe_allow_html=True,
    )

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{
            "role": "assistant",
            "content": (
                "Hi, I'm **Neuron** — your mental health support companion.\n\n"
                "I'm here to listen and offer evidence-based coping strategies. "
                "You can talk to me about anxiety, stress, sleep issues, relationships, "
                "or anything that's been on your mind.\n\n"
                "**How are you feeling today?**"
            ),
        }]

    # Render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f"<div class='bubble-label' style='text-align:right'>You</div>"
                f"<div class='bubble-user'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='bubble-label'>🧠 Neuron</div>"
                f"<div class='bubble-bot'>{msg['content']}</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Input row
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        user_text = st.text_input(
            "msg", label_visibility="collapsed",
            placeholder="Type your message here...",
            key="user_input",
        )
    with col_btn:
        send_btn = st.button("Send →", use_container_width=True)

    # Quick prompts
    st.markdown("<p style='color:#1a5c42;font-size:.82rem;margin:10px 0 4px;font-weight:700;'>Quick prompts:</p>", unsafe_allow_html=True)
    qc1, qc2, qc3, qc4 = st.columns(4)
    quick_prompts = {
        "😰 Feeling anxious":      "I've been feeling really anxious lately and I don't know why",
        "😴 Can't sleep":          "I've been struggling to fall asleep and feel exhausted all the time",
        "😞 Feeling low":          "I've been feeling really low and unmotivated recently",
        "💼 Work stress":          "Work stress is overwhelming me and I don't know how to cope",
    }
    triggered_prompt = None
    for col, (label, prompt) in zip([qc1, qc2, qc3, qc4], quick_prompts.items()):
        with col:
            if st.button(label, use_container_width=True, key=f"qp_{label}"):
                triggered_prompt = prompt

    final_text = triggered_prompt or (user_text if send_btn else None)

    if final_text and final_text.strip():
        st.session_state.messages.append({"role": "user", "content": final_text})

        history = [m for m in st.session_state.messages if m["role"] in ("user", "assistant")]

        with st.spinner("Neuron is thinking..."):
            full = ""
            for chunk in chatbot.stream_response(history, model=selected_model):
                full += chunk

        if chatbot.is_crisis_message(final_text):
            full += chatbot.CRISIS_ADDENDUM

        st.session_state.messages.append({"role": "assistant", "content": full})
        st.rerun()

    # Clear button
    if st.button("🗑️ Clear conversation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hi again! How are you feeling? I'm here to listen. 😊",
        }]
        st.rerun()

# TAB 2 ── RISK ASSESSMENT  (uses ml_model.py)
with tab2:
    st.markdown("### 📋 Mental Health Risk Assessment")
    st.markdown(
        "<p style='color:#1a5c42;margin-top:-8px;font-weight:600;'>"
        "Answer a few questions to get a personalised risk evaluation "
        "powered by a Random Forest model trained on clinical survey data.</p>",
        unsafe_allow_html=True,
    )

    with st.form("assessment_form"):
        st.markdown("#### 👤 About You")
        c1, c2 = st.columns(2)
        with c1:
            f_age    = st.slider("Age", 18, 70, 25)
            f_gender = st.selectbox("Gender", ["Male", "Female", "Non-binary", "Prefer not to say"])
        with c2:
            f_work   = st.slider("Work / Study hours per week", 10, 80, 40)
            f_screen = st.slider("Daily screen time (hours)", 1, 14, 5)

        st.markdown("#### 🌙 Lifestyle")
        c3, c4 = st.columns(2)
        with c3:
            f_sleep    = st.slider("Average sleep hours per night", 3.0, 10.0, 7.0, 0.5)
            f_exercise = st.slider("Exercise days per week", 0, 7, 2)
        with c4:
            f_stress  = st.slider("Stress level  (1 = very low · 10 = very high)", 1, 10, 5)
            f_social  = st.slider("Social support (1 = isolated · 10 = very supported)", 1, 10, 6)

        st.markdown("#### 📝 Mood Check  (last 2 weeks)")
        st.markdown(
            "<p style='color:#1a5c42;font-size:.83rem;font-weight:600;'>"
            "0 = Not at all &nbsp;·&nbsp; 1 = Several days &nbsp;·&nbsp; "
            "2 = More than half the days &nbsp;·&nbsp; 3 = Nearly every day</p>",
            unsafe_allow_html=True,
        )
        q1 = st.select_slider("Little interest or pleasure in things you usually enjoy",   options=[0,1,2,3], value=0)
        q2 = st.select_slider("Feeling down, depressed, or hopeless",                      options=[0,1,2,3], value=0)
        q3 = st.select_slider("Trouble sleeping or sleeping too much",                     options=[0,1,2,3], value=0)
        q4 = st.select_slider("Feeling tired or having very little energy",                options=[0,1,2,3], value=0)
        q5 = st.select_slider("Feeling nervous, anxious, or on edge",                      options=[0,1,2,3], value=0)

        submitted = st.form_submit_button("🔍  Assess My Risk", use_container_width=True)

    if submitted:
        phq_bonus  = (q1 + q2 + q3 + q4) * 0.4
        adj_stress = min(10, f_stress + phq_bonus)
        adj_sleep  = max(3,  f_sleep  - q3 * 0.4)

        risk_label, probs = ml_model.predict(
            age=f_age, sleep_hours=adj_sleep, stress_level=adj_stress,
            exercise_days=f_exercise, social_support=f_social,
            work_hours=f_work, screen_time=f_screen,
        )
        risk_icon = {"Minimal":"🟢","Mild":"🟡","Moderate":"🟠","Severe":"🔴"}[risk_label]

        st.markdown("---")
        st.markdown("### 📊 Your Results")

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Risk Level",    f"{risk_icon} {risk_label}")
        r2.metric("Stress",        f"{f_stress}/10")
        r3.metric("Sleep",         f"{f_sleep} hrs")
        r4.metric("PHQ Hint Score",f"{q1+q2+q3+q4}/12")

        st.markdown(
            f"<div class='card'>"
            f"<b style='color:#1a5c42;font-size:.85rem;text-transform:uppercase;letter-spacing:.05em;'>Predicted Risk Level</b><br><br>"
            f"<span class='badge badge-{risk_label}'>{risk_icon} {risk_label} Risk</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Probability chart
        fig_p = go.Figure(go.Bar(
            x=list(probs.keys()), y=list(probs.values()),
            marker_color=["#22c55e","#eab308","#f97316","#ef4444"],
            text=[f"{v}%" for v in probs.values()], textposition="outside",
            textfont=dict(color="#1a2744"),
        ))
        fig_p.update_layout(
            title="Probability Distribution Across Risk Classes",
            yaxis_title="Probability (%)", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(210,240,232,0.5)",
            font=dict(color="#1a2744", family="Nunito"),
            title_font=dict(color="#1a5c42", size=14),
            margin=dict(t=45,b=30,l=35,r=15),
            xaxis=dict(gridcolor="#b8ddd0"),
            yaxis=dict(gridcolor="#b8ddd0"),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Recommendations
        RECS = {
            "Minimal": [
                "✅ You're doing great — keep up those healthy habits.",
                "🧘 Continue regular exercise and a consistent sleep schedule.",
                "🤝 Nurture your social connections; they're a strong protective factor.",
                "📖 Journaling or mindfulness can help maintain your positive baseline.",
            ],
            "Mild": [
                "🌙 Aim for 7–9 hours of sleep and keep a consistent bedtime.",
                "🏃 Try to exercise 3–4 days per week — even a 20-min walk counts.",
                "🧘 Practice 10 minutes of mindfulness or deep breathing daily.",
                "💬 Open up to a trusted friend, family member, or mentor.",
                "📵 Reduce blue-light screen time at least 1 hour before bed.",
            ],
            "Moderate": [
                "⚠️ Consider speaking with a mental health professional soon.",
                "📅 Build a structured daily routine — predictability reduces anxiety.",
                "🚫 Limit caffeine, alcohol, and excessive social media scrolling.",
                "🧠 Try CBT-based techniques: thought journaling, behavioural activation.",
                "🆘 iCall India: 9152987821  |  Vandrevala: 1860-2662-345",
            ],
            "Severe": [
                "🚨 Please seek professional mental health support as soon as possible.",
                "📞 Crisis helpline: Vandrevala Foundation — 1860-2662-345 (24/7)",
                "🏥 Visit a psychiatrist, psychologist, or your nearest health centre.",
                "👨‍👩‍👧 Inform a trusted person about how you are feeling right now.",
                "🛑 Avoid making major life decisions until you've spoken with a professional.",
            ],
        }
        st.markdown("#### 💡 Personalised Recommendations")
        for rec in RECS[risk_label]:
            st.markdown(f"- {rec}")

        st.info("💬 Want to talk through these results? Head to the **Chat Support** tab.")

# TAB 3 ── DATA INSIGHTS  (uses eda.py)
with tab3:
    st.markdown("### 📊 Exploratory Data Analysis")
    st.markdown(
        "<p style='color:#1a5c42;margin-top:-8px;font-weight:600;'>"
        "Insights derived from 1,000 synthetic mental health survey records "
        "modelled on PHQ-9 and GAD-7 clinical patterns.</p>",
        unsafe_allow_html=True,
    )

    df    = load_dataset()
    stats = eda.summary_stats(df)

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Records",       f"{stats['total']:,}")
    k2.metric("Avg PHQ-9",     stats["avg_phq9"])
    k3.metric("Avg GAD-7",     stats["avg_gad7"])
    k4.metric("Avg Stress",    f"{stats['avg_stress']}/10")
    k5.metric("Avg Sleep",     f"{stats['avg_sleep']} hrs")
    k6.metric("High Risk %",   f"{stats['high_risk_pct']}%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Row 1
    col_a, col_b = st.columns(2)
    with col_a: st.plotly_chart(eda.chart_risk_distribution(df),  use_container_width=True)
    with col_b: st.plotly_chart(eda.chart_gender_risk(df),        use_container_width=True)

    # Row 2
    col_c, col_d = st.columns(2)
    with col_c: st.plotly_chart(eda.chart_sleep_vs_phq(df),       use_container_width=True)
    with col_d: st.plotly_chart(eda.chart_stress_vs_exercise(df), use_container_width=True)

    # Row 3
    col_e, col_f = st.columns(2)
    with col_e: st.plotly_chart(eda.chart_age_by_risk(df),        use_container_width=True)
    with col_f: st.plotly_chart(eda.chart_phq_vs_gad(df),         use_container_width=True)

    # Row 4 — full width
    st.plotly_chart(eda.chart_correlation_heatmap(df), use_container_width=True)
    st.plotly_chart(eda.chart_work_hours_dist(df),     use_container_width=True)

    with st.expander("🗃️ Browse raw dataset"):
        st.dataframe(df.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download full CSV",
            df.to_csv(index=False).encode(),
            "mental_health_dataset.csv", "text/csv",
        )

# TAB 4 ── ML MODEL INFO  (uses ml_model.py)
with tab4:
    st.markdown("### 🤖 ML Model Insights")
    st.markdown(
        "<p style='color:#1a5c42;margin-top:-8px;font-weight:600;'>"
        "Random Forest Classifier — trained to predict mental health risk level "
        "from lifestyle and symptom features.</p>",
        unsafe_allow_html=True,
    )

    model_obj, scaler_obj, features_list = load_ml_model()

    # Model cards
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Algorithm",  "Random Forest")
    m2.metric("Estimators", "200 trees")
    m3.metric("Classes",    "4 risk levels")
    m4.metric("Test split", "20 %")

    # Feature importance
    fi_pairs = ml_model.get_feature_importance()
    fi_names  = [p[0].replace("_", " ").title() for p in fi_pairs]
    fi_values = [round(p[1], 4) for p in fi_pairs]

    fig_fi = go.Figure(go.Bar(
        x=fi_values, y=fi_names, orientation="h",
        marker=dict(
            color=fi_values,
            colorscale="Purples",
            showscale=False,
        ),
        text=[f"{v:.3f}" for v in fi_values], textposition="outside",
        textfont=dict(color="#2d1b4e"),
    ))
    fig_fi.update_layout(
        title="Feature Importance (Random Forest)",
        xaxis_title="Importance Score", height=360,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(210,240,232,0.5)",
        font=dict(color="#1a2744", family="Nunito"),
        title_font=dict(color="#1a5c42", size=14),
        margin=dict(t=45,b=30,l=160,r=60),
        xaxis=dict(gridcolor="#b8ddd0"),
        yaxis=dict(gridcolor="#b8ddd0"),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Confusion matrix + accuracy
    from sklearn.metrics import confusion_matrix
    y_test, y_pred, acc = ml_model.get_eval_data()
    cm = confusion_matrix(y_test, y_pred)
    RISK_NAMES = ["Minimal","Mild","Moderate","Severe"]

    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=RISK_NAMES, y=RISK_NAMES,
        colorscale="Purples",
        text=cm, texttemplate="%{text}",
        textfont=dict(size=14, color="#2d1b4e"),
    ))
    fig_cm.update_layout(
        title=f"Confusion Matrix  (Test Accuracy: {acc:.1%})",
        xaxis_title="Predicted", yaxis_title="Actual", height=400,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(210,240,232,0.5)",
        font=dict(color="#1a2744", family="Nunito"),
        title_font=dict(color="#1a5c42", size=14),
    )

    col_cm, col_summary = st.columns([3, 2])
    with col_cm:
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_summary:
        st.markdown(f"""
        <div class='card' style='font-size:.88rem;line-height:2;'>
        <b style='color:#1a5c42;font-family:Playfair Display,serif;font-size:1.05rem;'>Model Summary</b><br>
        <b style='color:#2a6858;'>Algorithm:</b> <span style='color:#0f2d20;'>Random Forest</span><br>
        <b style='color:#2a6858;'>Test Accuracy:</b> <span style='color:#0f2d20;'>{acc:.1%}</span><br>
        <b style='color:#2a6858;'>Features:</b> <span style='color:#0f2d20;'>{len(features_list)}</span><br>
        <b style='color:#2a6858;'>Training rows:</b> <span style='color:#0f2d20;'>800</span><br>
        <b style='color:#2a6858;'>Test rows:</b> <span style='color:#0f2d20;'>200</span><br>
        <b style='color:#2a6858;'>Class weighting:</b> <span style='color:#0f2d20;'>Balanced</span><br>
        <b style='color:#2a6858;'>Scaler:</b> <span style='color:#0f2d20;'>StandardScaler</span><br>
        </div>
        """, unsafe_allow_html=True)

    # Pipeline
    st.markdown("#### 🔄 Prediction Pipeline")
    st.markdown("""
    <div class='card' style='display:flex;gap:10px;flex-wrap:wrap;align-items:center;'>
        <div style='background:#d4f0e8;border:2px solid #2a9d8f;
                    border-radius:8px;padding:8px 16px;color:#0f3d2e;font-weight:800;'>
            📋 Assessment Form
        </div><span style='color:#2a9d8f;font-weight:800;font-size:1.1rem;'>→</span>
        <div style='background:#d4f0e8;border:2px solid #2a9d8f;
                    border-radius:8px;padding:8px 16px;color:#0f3d2e;font-weight:800;'>
            🔧 StandardScaler
        </div><span style='color:#2a9d8f;font-weight:800;font-size:1.1rem;'>→</span>
        <div style='background:#d4f0e8;border:2px solid #2a9d8f;
                    border-radius:8px;padding:8px 16px;color:#0f3d2e;font-weight:800;'>
            🌲 Random Forest (200 trees)
        </div><span style='color:#2a9d8f;font-weight:800;font-size:1.1rem;'>→</span>
        <div style='background:#d4f0e8;border:2px solid #2a9d8f;
                    border-radius:8px;padding:8px 16px;color:#0f3d2e;font-weight:800;'>
            🎯 Risk Level + Probabilities
        </div><span style='color:#2a9d8f;font-weight:800;font-size:1.1rem;'>→</span>
        <div style='background:#d4f0e8;border:2px solid #2a9d8f;
                    border-radius:8px;padding:8px 16px;color:#0f3d2e;font-weight:800;'>
            💬 Chatbot + Recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)