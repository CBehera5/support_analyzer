# 🧠 Support Ticket Analyzer – Doc or UI Fix?

This Streamlit web app uses the **Groq API** and a **vector search-based RAG system** to analyze customer support tickets. It determines whether the issue is due to missing/unclear documentation, a poor UI/UX design, or both.

> ✨ Built using **Streamlit**, **Groq API (LLaMA 3)**, **Chroma DB**, and **LangChain**.

---

1. Create Virtual Environment (Optional)
```
python -m venv venv
source venv/bin/activate  
```
2. Install Dependencies
```
pip install -r requirements.txt
```
3. Run the App
```
streamlit run app.py
```

## 🚀 Features

- Upload your support tickets (CSV)
- Upload your product documentation (TXT)
- Uses **semantic similarity** to fetch relevant docs
- Queries **Groq’s LLaMA 3 model** for smart recommendations
- Suggests: `Update Documentation`, `Improve UI`, or `Both`
- Download results as a CSV