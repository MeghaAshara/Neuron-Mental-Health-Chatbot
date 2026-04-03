# 🧠 Neuron — Mental Health Analytics Platform

## 📁 Project Structure
```
Neuron/
│
├── data/
│   ├── generate_dataset.py       ← Generates 1,000-row PHQ-9/GAD-7 dataset
│   └── mental_health_dataset.csv ← Auto-created on setup
│
├── eda.py          ← All EDA chart functions (Plotly)
├── ml_model.py     ← Random Forest training, prediction, evaluation
├── chatbot.py      ← Ollama chatbot with streaming + crisis detection
│
├── app.py          ← Main Streamlit app (4 tabs, imports the above 3)
├── setup.py        ← One-time setup script
└── requirements.txt
```

## 🚀 Setup & Run
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup Ollama
ollama serve          # in one terminal
ollama pull llama3.2  # pull a model

# 3. Generate data + train model (run ONCE)
python setup.py

# 4. Launch
streamlit run app.py
# Open http://localhost:8501
```

## 📱 App Tabs

| Tab | Module | What it does |
|---|---|---|
| 💬 Chat Support | `chatbot.py` | Streaming Ollama chatbot, quick prompts, crisis detection |
| 📋 Risk Assessment | `ml_model.py` | PHQ-9 intake form → ML prediction → recommendations |
| 📊 Data Insights | `eda.py` | 8 interactive EDA charts + KPIs + raw data download |
| 🤖 ML Model Info | `ml_model.py` | Feature importance, confusion matrix, pipeline diagram, GTU checklist |

## 🔧 Extending with Deep Learning

```bash
pip install transformers torch
```
```python
# In chatbot.py — detect emotion from user messages
from transformers import pipeline
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)
result = emotion_classifier("I feel so hopeless and tired")
# → [{'label': 'sadness', 'score': 0.92}]
```

---

## ⚠️ Disclaimer
Educational purposes only. Not a substitute for professional medical care.

**Crisis Helplines (India):**
- iCall: 9152987821
- Vandrevala Foundation: 1860-2662-345
- SNEHI: 044-24640050
