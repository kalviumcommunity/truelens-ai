# 🔍 TrueLens AI

**TrueLens AI** is an AI-powered authenticity checker that detects whether text is **AI-generated** or **human-written** — and explains *why*.  
It goes beyond simple classification by providing reasoning, confidence scores, and highlighted annotations for suspicious phrases.

---

## 🧠 Features

- 🆚 **Multi-Mode Detection** — Zero-shot, One-shot, and Multi-shot classification.
- 🧩 **Chain-of-Thought Reasoning** — Transparent explanations for each decision.
- 📝 **Annotated Highlights** — Marks AI-likely phrases directly in the text.
- ⚙️ **Parameter Controls** — Adjust temperature, top-p, and top-k.
- 📊 **Evaluation Mode** — Test accuracy on labeled datasets with metrics (Accuracy, Precision, Recall, F1-score).
- 💾 **Embeddings Support** — Compare content similarity with AI and human reference samples.

---

## 🛠️ Tech Stack

| Area        | Tech                                       |
|-------------|--------------------------------------------|
| Frontend    | Streamlit (Python UI)                      |
| Backend     | Python, OpenAI API / Hugging Face Inference |
| ML          | GPT-4o-mini or LLaMA3 for detection         |
| Embeddings  | Sentence Transformers + ChromaDB           |
| Storage     | CSV datasets / ChromaDB persistence        |
| Hosting     | Streamlit Cloud                            |

---

## 📁 Project Structure
```
truelens-ai/
├── app.py         # Streamlit main UI
├── detection/     # Detection logic & prompt templates
├── evaluation/    # Dataset evaluation scripts
├── embeddings/    # Embedding storage & similarity search
├── data/          # Sample labeled datasets
└── README.md
```

---

## ⚙️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 2️⃣ Run the App
```bash
streamlit run app.py
```

---

## ✅ Sample API Usage

**Input**

```text
As an AI language model, I am here to help you with your request.
```

**Response**
```json
{
  "classification": "AI-generated",
  "confidence": 0.94,
  "reasoning": "Phrases like 'As an AI language model' are direct AI indicators.",
  "annotations": [
    { "text": "As an AI language model", "type": "AI-likely" }
  ]
}
```
---

## 🔐 Security Focus

- API keys stored securely in \`.env\`
- Parameter limits to prevent misuse
- Optional offline embeddings mode for sensitive datasets

---

## 📌 Future Roadmap

- [ ]  Add browser extension for inline detection
- [ ] Expand dataset for domain-specific AI text
- [ ] Support for audio & image caption authenticity
- [ ] Deploy with Docker for reproducibility

---

> Built by **Jessica Agarwal**  
> [GitHub](https://github.com/jessicaagarwal) | [LinkedIn](https://www.linkedin.com/in/jessica-agarwal-00b6b7225/)
