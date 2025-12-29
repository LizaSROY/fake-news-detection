import streamlit as st
import torch
import re
from transformers import RobertaTokenizer, RobertaForSequenceClassification


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)
@st.cache_resource
def load_model():
    tokenizer = RobertaTokenizer.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
    model = RobertaForSequenceClassification.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

st.title("📰 Fake News Detection System")
st.markdown(
    "This system uses a **RoBERTa-based deep learning model** to classify news articles as **Fake** or **Real**."
)
st.divider()
news_text = st.text_area(
    "Enter news article text:",
    height=250,
    placeholder="Paste the full news article here..."
)
col1, col2 = st.columns(2)

with col1:
    predict_btn = st.button("🔍 Predict", use_container_width=True)

with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
if clear_btn:
    st.rerun()

if predict_btn:
    if not news_text.strip():
        st.warning("⚠️ Please enter a news article before predicting.")
    else:
        with st.spinner("Analyzing article..."):
            cleaned_text = clean_text(news_text)

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

            label = torch.argmax(probs).item()
            confidence = probs[label].item()


# @st.cache_resource
# def load_model():
#     model = RobertaForSequenceClassification.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     tokenizer = RobertaTokenizer.from_pretrained("/Users/macbook/Desktop/fake_real/roberta_model")
#     model.eval()
#     return model, tokenizer

# model, tokenizer = load_model()

# def clean_text(text):
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"<.*?>", "", text)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

# st.title("📰 Fake News Detection (RoBERTa)")

# text = st.text_area("Enter news article")

# if st.button("Predict"):
#     text = clean_text(text)

#     inputs = tokenizer(
#         text,
#         padding="max_length",
#         truncation=True,
#         max_length=512,
#         return_tensors="pt"
#     )

#     with torch.no_grad():
#         outputs = model(**inputs)
#         probs = torch.softmax(outputs.logits, dim=1)[0]

#     label = torch.argmax(probs).item()
#     confidence = probs[label].item()

#     st.write("Prediction:", "🟥 Fake" if label == 0 else "🟩 Real")
#     st.write(f"Confidence: {confidence*100:.2f}%")
