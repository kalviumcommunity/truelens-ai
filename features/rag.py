# features/rag.py
import os
import json
from groq import Groq
from utils.config import GROQ_API_KEY, GROQ_MODEL
from core.embeddings import get_embeddings
from core.vectorstore import build_faiss_index, load_faiss_index, search
import numpy as np

SAMPLE_DB = [
    # authentic samples
    {"text": "I took this photo during my trip to Paris in the summer of 2019.", "label": "Authentic"},
    {"text": "The conference was held in Mumbai and featured over 300 local entrepreneurs.", "label": "Authentic"},
    {"text": "We celebrated my sister's birthday at her favorite cafe last Friday.", "label": "Authentic"},
    # AI samples
    {"text": "This picturesque scene captures the ethereal charm of a sun-kissed Parisian morning.", "label": "AI-generated"},
    {"text": "In a remarkable convergence of minds, the Mumbai conference unveiled unprecedented entrepreneurial synergy.", "label": "AI-generated"},
    {"text": "The versatile framework enables seamless horizontal scaling and frictionless orchestration.", "label": "AI-generated"},
]

def ensure_index():
    idx, metas = load_faiss_index()
    if idx is not None:
        return
    # build index
    texts = [d["text"] for d in SAMPLE_DB]
    embs = get_embeddings(texts)  # (n, dim)
    metadatas = [{"text": d["text"], "label": d["label"]} for d in SAMPLE_DB]
    build_faiss_index(embs, metadatas)
    print("Built FAISS index with sample DB.")

def run_query_with_rag(query_text: str, top_k: int = 4):
    ensure_index()
    # embed query
    q_emb = get_embeddings(query_text)  # vector
    # retrieve
    retrieved = search(q_emb, top_k=top_k)
    # build context string
    context_lines = []
    for r in retrieved:
        md = r["metadata"]
        context_lines.append(f"- [{md.get('label')}] {md.get('text')} (score: {r['score']:.4f})")
    retrieved_text_block = "\n".join(context_lines)

    # compose prompt to Groq
    prompt = (
        "You are an AI authenticity detection expert. You will be given a short query text and "
        "several contextual examples labelled as Authentic or AI-generated. Use the examples to help classify the query. "
        "Return a JSON with fields: classification (Authentic or AI-generated), confidence (0-1), reasoning (step-by-step), retrieved_examples (list).\n\n"
        f"CONTEXT EXAMPLES:\n{retrieved_text_block}\n\n"
        f"QUERY:\n{query_text}\n\n"
        "Respond ONLY with valid JSON."
    )

    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful text authenticity classifier."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=400
    )
    # Groq SDK returns choice.message as object
    raw = resp.choices[0].message.content.strip()
    # try to parse JSON, but model might include extra text -> handle gracefully
    parsed = None
    try:
        parsed = json.loads(raw)
    except Exception:
        # attempt to extract JSON substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw[start:end+1])
            except Exception:
                parsed = {"raw_output": raw}
        else:
            parsed = {"raw_output": raw}

    result = {
        "query": query_text,
        "retrieved": retrieved,
        "model_raw": raw,
        "parsed": parsed
    }
    return result

def run():
    # interactive demo with a few queries
    examples = [
        "I took this photo during my trip to Paris in the summer of 2019.",
        "The versatile framework enables seamless horizontal scaling and frictionless orchestration.",
        "The conference was held in Mumbai and featured over 300 local entrepreneurs."
    ]
    for q in examples:
        print("\n===== QUERY =====")
        print(q)
        out = run_query_with_rag(q, top_k=3)
        print("\n=== RAG RESULT ===")
        print(json.dumps(out["parsed"], indent=2))
        print("\n( Raw model output )\n", out["model_raw"])