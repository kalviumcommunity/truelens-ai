from sentence_transformers import SentenceTransformer, util
from utils.config import EMBEDDING_MODEL

def run():
    model = SentenceTransformer(EMBEDDING_MODEL)

    authentic_samples = [
        "I took this photo during my trip to Paris in the summer of 2019.",
        "The conference was held in Mumbai and featured over 300 local entrepreneurs."
    ]

    ai_generated_samples = [
        "This picturesque scene captures the ethereal charm of a sun-kissed Parisian morning.",
        "In a remarkable convergence of minds, the Mumbai conference unveiled unprecedented entrepreneurial synergy."
    ]

    auth_embeddings = model.encode(authentic_samples, convert_to_tensor=True)
    ai_embeddings = model.encode(ai_generated_samples, convert_to_tensor=True)

    print("\n=== Cosine Similarities Between Authentic & AI-Generated ===")
    for i, auth_text in enumerate(authentic_samples):
        for j, ai_text in enumerate(ai_generated_samples):
            similarity = util.cos_sim(auth_embeddings[i], ai_embeddings[j]).item()
            print(f"\nAuthentic: {auth_text}\nAI-Generated: {ai_text}\nSimilarity: {similarity:.4f}")

    print("\nEmbedding Vector Size:", auth_embeddings.shape[1])