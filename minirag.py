"""
minirag.py — A minimal Retrieval-Augmented Generation (RAG) implementation using SentenceTransformers and FAISS.
"""

import os
from typing import List, Tuple

# -----------------------------
# Dependencies
# -----------------------------
# pip install sentence-transformers faiss-cpu openai

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError("Install sentence-transformers:\n  pip install sentence-transformers")

try:
    import faiss
except Exception as e:
    raise ImportError("Install FAISS:\n  pip install faiss-cpu")

# Optional (only needed for LLM generation)
try:
    import openai
    OPENAI_AVAILABLE = True
except:
    OPENAI_AVAILABLE = False


# -----------------------------
# Documents for RAG
# -----------------------------
DOCS = [
    {
        "id": "masala_dosa",
        "title": "Masala Dosa Recipe",
        "text": (
            "# Masala Dosa\n"
            "Ingredients: dosa batter, potatoes, onion, turmeric, curry leaves.\n"
            "Method: Ferment batter overnight. Prepare potato masala. Spread dosa batter on tawa, "
            "add masala, fold and serve hot."
        )
    },
    {
        "id": "idli_recipe",
        "title": "Idli Recipe",
        "text": (
            "# Idli\n"
            "Ingredients: idli rice, urad dal, salt.\n"
            "Method: Soak and grind rice & dal separately. Mix, ferment overnight, steam in idli molds."
        )
    },
    {
        "id": "HYDERABADI _CHICKEN_BIRYANI",
        "title": "HYDERABADI CHICKEN BIRYANI",
        "text": (
            "# HYDERABADI CHICKEN BIRYANI\n"
            "Ingredients: 1 kg chicken,1 cup curd,2 tbsp ginger-garlic paste,2 tsp red chili powder,2 tsp garam masala."
            
        )
    },
    {
        "id": "MeduVada",
        "title": "Medu Vada Recipe",
        "text": (
            "# Medu Vada\n"
            "Ingredients: urad dal, green chilies, ginger, curry leaves, salt.\n"
            "Method: Soak and grind urad dal. Mix in spices, shape into vadas, deep fry until golden."

        )
    }
]


# -----------------------------
# Build Embeddings
# -----------------------------
def build_embeddings(docs: List[dict], model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    texts = [d["text"] for d in docs]
    print(f"[mini-rag] Encoding {len(texts)} documents...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    ids = [d["id"] for d in docs]
    return ids, embeddings


# -----------------------------
# Build FAISS Index
# -----------------------------
def build_faiss_index(embeddings, metric="cosine"):
    dim = embeddings.shape[1]

    if metric == "cosine":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    index.add(embeddings)
    return index


# -----------------------------
# Retrieve
# -----------------------------
def retrieve(query: str, docs: List[dict], index, model, top_k=5, min_score=0.30):
    """
    Retrieve documents using FAISS but filter by semantic relevance.
    Returns only docs whose similarity >= min_score.
    """
    q_emb = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)

    # retrieve more results than needed
    D, I = index.search(q_emb, top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        print (score)
        if score >= min_score:    # KEEP only relevant docs
            doc = docs[idx].copy()
            doc["_score"] = float(score)
            results.append(doc)

    # If no document is above threshold → return empty list
    return results



# -----------------------------
# Prompt Builder
# -----------------------------
def build_prompt(question: str, retrieved_docs: List[dict]) -> str:
    buf = []
    buf.append("You are an AI assistant. Use the retrieved context to answer the user's question.\n")

    buf.append("### Retrieved Context ###\n")
    for d in retrieved_docs:
        buf.append(f"TITLE: {d['title']}\n{d['text']}\n\n")

    buf.append("### Question ###\n")
    buf.append(question + "\n\n")

    buf.append("### Answer ###\n")
    return "".join(buf)


# -----------------------------
# LLM Generation (optional)
# -----------------------------
def generate_with_openai(prompt: str):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI package not installed.\nInstall: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable.")

    openai.api_key = api_key

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
    )
    return response["choices"][0]["message"]["content"]


# -----------------------------
# PUBLIC API: get_answer()
# -----------------------------
def get_answer(question: str, top_k: int = 5):
    global DOCS, _model, _index

    retrieved = retrieve(question, DOCS, _index, _model, top_k=top_k, min_score=0.30)

    if not retrieved:
        return [], "No relevant knowledge found in the RAG documents."

    prompt = build_prompt(question, retrieved)

    if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            answer = generate_with_openai(prompt)
        except Exception as e:
            answer = f"[OpenAI Error] {e}"
    else:
        answer = "\n".join([d["text"] for d in retrieved])

    return retrieved, answer



# -----------------------------
# Initialize globals on import
# -----------------------------
print("[mini-rag] Initializing...")

ids, _emb = build_embeddings(DOCS)
_index = build_faiss_index(_emb, metric="cosine")
_model = SentenceTransformer("all-MiniLM-L6-v2")

print("[mini-rag] Ready.")


# -----------------------------
# Run demo if executed directly
# -----------------------------
if __name__ == "__main__":
    print("\n--- MINI RAG DEMO ---\n")
    q = "How do I make Idli?"
    retrieved, answer = get_answer(q)

    print("Query:", q)
    print("\nRetrieved Docs:")
    for r in retrieved:
        print(f" - {r['id']} (score={r['_score']:.4f})")

    print("\nAnswer:\n", answer)
