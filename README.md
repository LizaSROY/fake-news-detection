📰 Fake News Detection using NLP & Deep Learning

📌 Overview

**Fake news has become a major challenge in the digital era, spreading misinformation
rapidly through social media and online platforms.
This project aims to automatically classify news articles as Fake or Real
using Natural Language Processing (NLP) and deep learning techniques.**

**The system is built using a transformer-based model and deployed as an
interactive Streamlit web application.**

### Project Objectives

  * Detect fake news based on textual content
  
  * Apply NLP preprocessing techniques
  
  * Fine-tune a transformer-based deep learning model
  
  * Deploy the trained model in a user-friendly web interface

### Model

  * Architecture: Transformer-based model (RoBERTa)
  
  * Task: Binary text classification (Fake / Real)
  
  * Framework: PyTorch + Hugging Face Transformers
  
  * Model Hosting: Hugging Face (model size exceeds GitHub limits)

### Pretrained Model:
  ** [sroyliza/LizaSR-fake-news-roberta](https://huggingface.co/sroyliza/LizaSR-fake-news-roberta) **

### Project Structure

  fake-news-detection/
  │
  ├── main.ipynb        # Data preprocessing, training, and evaluation
  ├── app.py            # Streamlit web application
  ├── requirements.txt  # Project dependencies
  ├── .gitignore        # Ignored files and folders
  ├── LICENSE           # MIT License
  └── README.md         # Project documentation


### Dataset

  * Public fake news dataset (Kaggle)
  
  * Dataset has 44898 rows and 6 columns, combined between fake and real news
  
  * Contains labeled news articles (Fake / Real)
  
  * Text data cleaned and preprocessed before training
  
  * Dataset is not included due to size limitations


### Installation & Setup

  ## Clone the Repository
    -> git clone https://github.com/LizaSROY/fake-news-detection.git
    -> cd fake-news-detection

  ## Install Dependencies
    -> pip install -r requirements.txt

  ## Run the Application
    -> streamlit run app.py

### Model Performance

  The model achieved strong performance on the test dataset:

  * Accuracy: 100%
    
  * Precision: 1.00
    
  * Recall: 1.00
    
  * F1-score: 1.00
    
  * ROC–AUC: 0.9988

**These results indicate that the model is able to reliably distinguish between fake and real news with very high confidence.**

### Technologies Used

  * Programming Language: Python
  
  * Deep Learning: PyTorch
  
  * NLP Models: Hugging Face Transformers
  
  * Web App: Streamlit
  
  * Data Processing: Pandas, NumPy
    
  * Evaluation: Scikit-learn

### Deployment

  * Model hosted on Hugging Face
  
  * Web application built with Streamlit
  
  * GitHub repository contains full source code (excluding large model files)

### Notes

  * Trained model weights are not stored in this repository
  
  * Large model files are hosted externally for efficiency
  
  * No sensitive data or credentials are included


👤 Author: Liza SROY

  -> Linkedin: ** www.linkedin.com/in/lizasroy99 **
  -> Majors:  Data Science & AI Engineering

⭐ Acknowledgements

  * Hugging Face Transformers
  
  * Streamlit
  
  * Kaggle Datasets








