
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk

nltk.download('punkt')


def summarize_text(text, n_sentences=3):
    sentences = nltk.sent_tokenize(text)

    if len(sentences) <= n_sentences:
        return text

 
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(sentences)

 
    similarity_matrix = cosine_similarity(X)

   
    graph = nx.from_numpy_array(similarity_matrix)

   
    scores = nx.pagerank(graph)

 
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

  
    selected_sentences = [ranked_sentences[i][1] for i in range(min(n_sentences, len(ranked_sentences)))]


    summary = " ".join(selected_sentences)
    return summary
