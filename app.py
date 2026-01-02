import streamlit as st
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, RobertaForSequenceClassification


st.set_page_config(
    page_title="Fake News Detection System",
    page_icon="📰",
    layout="wide"
)


st.markdown("""
<style>
.card {
    background-color: #f9fafb;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}
.title {
    font-size: 36px;
    font-weight: 700;
}
.subtitle {
    color: #6b7280;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = RobertaForSequenceClassification.from_pretrained(
        "roberta_model"
    )
    tokenizer = RobertaTokenizer.from_pretrained(
        "roberta_model"
    )
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clear_text():
    st.session_state.news_input = ""

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

if "user_history" not in st.session_state:
    st.session_state.user_history = {}

if "news_input" not in st.session_state:
    st.session_state.news_input = ""


if "users" not in st.session_state:

    st.session_state.users = {}

if "user_history" not in st.session_state:
    
    st.session_state.user_history = {}

if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "signup"


if not st.session_state.logged_in:

    st.markdown("""
    <style>
    .auth-card {
        background-color: #f9fafb;
        padding: 28px;
        border-radius: 14px;
        border: 1px solid #e5e7eb;
        max-width: 480px;
        margin-bottom: 24px;
    }
    .auth-title {
        font-size: 26px;
        font-weight: 700;
        margin-bottom: 4px;
    }
    .auth-subtitle {
        font-size: 14px;
        color: #6b7280;
        margin-bottom: 20px;
    }
    .auth-switch {
        font-size: 14px;
        color: #374151;
    }
    .auth-link {
        color: #2563eb;
        cursor: pointer;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='auth-card'>", unsafe_allow_html=True)

    
    if st.session_state.auth_mode == "signup":

        st.markdown("<div class='auth-title'>Create an account</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Join Fake News Detection to analyze news credibility</div>",
            unsafe_allow_html=True
        )

        new_username = st.text_input("Username")
        new_password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account", use_container_width=True):
            if not new_username or not new_password:
                st.warning("All fields are required")
            elif new_username in st.session_state.users:
                st.error("Username already exists")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                st.session_state.users[new_username] = new_password
                st.session_state.user_history[new_username] = []
                st.success("Account created successfully")
                st.session_state.auth_mode = "login"
                st.rerun()

        st.markdown(
            "<div class='auth-switch'>Already have an account? "
            "<span class='auth-link'>Login</span></div>",
            unsafe_allow_html=True
        )

        if st.button("Go to Login", key="to_login"):
            st.session_state.auth_mode = "login"
            st.rerun()

    else:

        st.markdown("<div class='auth-title'>Welcome back</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='auth-subtitle'>Log in to continue</div>",
            unsafe_allow_html=True
        )

        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if (
                username in st.session_state.users and
                st.session_state.users[username] == password
            ):
                st.session_state.logged_in = True
                st.session_state.username = username

                if username not in st.session_state.user_history:
                    st.session_state.user_history[username] = []

                st.rerun()
            else:
                st.error("Invalid username or password")

        st.markdown(
            "<div class='auth-switch'>Don’t have an account? "
            "<span class='auth-link'>Sign up</span></div>",
            unsafe_allow_html=True
        )

        if st.button("Go to Sign Up", key="to_signup"):
            st.session_state.auth_mode = "signup"
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()


st.sidebar.title("👤 User")
st.sidebar.write(f"**Logged in as:** {st.session_state.username}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

st.sidebar.divider()
st.sidebar.markdown("### ℹ️ About")
st.sidebar.write("""
This system uses **RoBERTa**, a transformer-based deep learning model fine-tuned for **Fake News Detection**, Designed for academic & real-world demo use.
""")

st.sidebar.markdown("### 🔍 Tips") 
st.sidebar.write(""" - Paste full news articles 
                     - Longer text gives better accuracy 
                     - Avoid headlines only """)

st.markdown("<div class='title'>📰 Fake News Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-powered news credibility analysis</div>", unsafe_allow_html=True)
st.divider()

st.markdown("<div class='card'>", unsafe_allow_html=True)

text = st.text_area(
    "📝 Enter News Article",
    height=220,
    key="news_input",
    placeholder="Paste full news article here..."
)

col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

with col2:
    st.button("🧹 Clear", use_container_width=True, on_click=clear_text)

st.markdown("</div>", unsafe_allow_html=True)


if predict_btn and text.strip():
    cleaned_text = clean_text(text)

    inputs = tokenizer(
        cleaned_text,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    fake_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100
    label = "Real" if real_prob > fake_prob else "Fake"

  
    st.session_state.user_history[st.session_state.username].append({
        "Text": cleaned_text[:60] + "...",
        "Fake (%)": round(fake_prob, 2),
        "Real (%)": round(real_prob, 2),
        "Prediction": label
    })


    st.subheader("📊 Prediction Result")

    if label == "Real":
        st.success(f"🟩 **Real News**")
    else:
        st.error(f"🟥 **Fake News**")

    st.write(f"**Fake Probability:** {fake_prob:.2f}%")
    st.progress(fake_prob / 100)

    st.write(f"**Real Probability:** {real_prob:.2f}%")
    st.progress(real_prob / 100)

  
    labels = ["Fake", "Real"]
    values = [fake_prob, real_prob]
    colors = ["#ef4444", "#22c55e"]

    fig, ax = plt.subplots(figsize=(5, 2.2))
    ax.barh(labels, values, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)")
    ax.set_title("Prediction Confidence", fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

user_hist = st.session_state.user_history.get(st.session_state.username, [])

if user_hist:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🕒 Your Prediction History")
    st.dataframe(pd.DataFrame(user_hist), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

