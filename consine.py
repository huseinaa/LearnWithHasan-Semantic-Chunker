import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

def calculate_cosine_similarities_sklearn(embeddings):
    # Calculate the cosine similarities between consecutive embeddings using sklearn.
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = sklearn_cosine_similarity(embeddings[i], embeddings[i + 1])[0][0]
        similarities.append(similarity)
    return similarities

def calculate_cosine_similarities_manual(embeddings):
    # Manually calculate the cosine similarities between consecutive embeddings.
    similarities = []
    for i in range(len(embeddings) - 1):
        vec1 = embeddings[i].flatten()
        vec2 = embeddings[i+1].flatten()
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        
        if norm_vec1 == 0 or norm_vec2 == 0:
            # If either vector is zero, similarity is undefined (could also return 0)
            similarity = float('nan')
        else:
            similarity = dot_product / (norm_vec1 * norm_vec2)
        similarities.append(similarity)
    return similarities

# Generate random embeddings as 2D single-row arrays
embeddings = [np.random.rand(1, 1536) for _ in range(10)]

# Calculate similarities using both functions
similarities_sklearn = calculate_cosine_similarities_sklearn(embeddings)
similarities_manual = calculate_cosine_similarities_manual(embeddings)

# Create a DataFrame to compare results
comparison_df = pd.DataFrame({
    'Pair Index': [f"{i}-{i+1}" for i in range(len(similarities_sklearn))],
    'Sklearn Similarity': similarities_sklearn,
    'Manual Similarity': similarities_manual
})

print(comparison_df)
