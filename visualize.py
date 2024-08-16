import re
import openai
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from SimplerLLM.tools.generic_loader import load_content

def _split_sentences(text):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    return sentences

def _combine_sentences(sentences):
    combined_sentences = []
    for i in range(len(sentences)):
        combined_sentence = sentences[i]
        if i > 0:
            combined_sentence = sentences[i-1] + ' ' + combined_sentence
        if i < len(sentences) - 1:
            combined_sentence += ' ' + sentences[i+1]
        combined_sentences.append(combined_sentence)
    return combined_sentences

def convert_to_vector(texts):
    try:
        response = openai.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        embeddings = np.array([item.embedding for item in response.data])
        return embeddings
    except Exception as e:
        print("An error occurred:", e)
        return np.array([])  # Return an empty array in case of an error

def calculate_cosine_similarities(embeddings):
    # Calculate the cosine similarities between consecutive embeddings.
    similarities = []
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0][0]
        similarities.append(similarity)
    return similarities

def plot_cosine_similarities(text):
    sentences = _split_sentences(text)
    combined_sentences = _combine_sentences(sentences)
    embeddings = convert_to_vector(combined_sentences)
    similarities = calculate_cosine_similarities(embeddings)
    
    plt.figure(figsize=(10, 5))
    # Plot the line connecting all points in blue
    plt.plot(similarities, marker='o', linestyle='-', color='blue', label='Cosine Similarity')
    
    # Overlay red dots where the similarity is 0.95 or higher
    for i, similarity in enumerate(similarities):
        if similarity >= 0.95:
            plt.plot(i, similarity, marker='o', color='red')
    
    plt.title('Cosine Similarities Between Consecutive Sentences')
    plt.xlabel('Sentence Pair Index')
    plt.ylabel('Cosine Similarity')
    plt.grid(True)
    plt.legend()
    plt.show()

# Example usage:
load = load_content("https://learnwithhasan.com/how-to-build-a-semantic-plagiarism-detector/")
text = load.content
plot_cosine_similarities(text)
