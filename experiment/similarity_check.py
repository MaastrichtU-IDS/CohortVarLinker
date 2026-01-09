from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from lazy_model import get_model

EMBEDDING_MODEL_NAME = "sapbert"

# sentences
sentences = [
    "heart attack;Cardiac infarction", "myocardial infarction;heart attack;Cardiac infarction", "high blood pressure"
]

def get_embedding_model(model_name=EMBEDDING_MODEL_NAME):
    return get_model(backend=model_name)

# check embeddings similarity
def compute_similarity(emb_model, sentences):
    embedding_model, embedding_size = get_embedding_model(model_name=emb_model)
    
    # 1. Process one by one to avoid the TypeError
    embeddings_list = []
    print(f"Embedding {len(sentences)} sentences one by one...")
    
    for sent in sentences:
        # embed_text expects a single string, so we pass 'sent' directly
        vector = embedding_model.embed_text(sent)
        embeddings_list.append(vector)
    
    # 2. Convert list of vectors to a numpy array (Shape: [3, 768])
    embeddings = np.array(embeddings_list)
    
    # 3. Compute Cosine Similarity Matrix
    sim_matrix = cosine_similarity(embeddings)
    return sim_matrix



# Run the computation
results = compute_similarity(EMBEDDING_MODEL_NAME, sentences)

# print("\n--- Similarity Matrix ---")
# print(results)

print("\n--- Pairwise Comparisons ---")
# Optional: Print them out legibly to compare them explicitly
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        score = results[i][j]
        print(f"'{sentences[i]}'  vs  '{sentences[j]}'")
        print(f"Score: {score:.4f}\n")