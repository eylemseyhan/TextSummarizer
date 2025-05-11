from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple, Dict
import string

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

# Download required NLTK resources silently
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

class TextPreprocessor:
    def __init__(self, language='english'):
        self.language = language
        # Custom Turkish stopwords
        self.turkish_stopwords = set([
            'acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz',
            'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem',
            'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu',
            'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki',
            've', 'veya', 'ya', 'yani'
        ])
        self.stopwords = set(stopwords.words(language)) if language == 'english' else self.turkish_stopwords

    def clean_text(self, text: str) -> str:
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text

    def remove_stopwords(self, text: str) -> str:
        words = word_tokenize(text)
        return ' '.join([w for w in words if w.lower() not in self.stopwords])

    def preprocess(self, text: str) -> str:
        return self.remove_stopwords(self.clean_text(text))

class TextSummarizer:
    _model_instance = None
    
    def __init__(self, language='english', use_transformers=True):
        self.language = language
        self.preprocessor = TextPreprocessor(language)
        self.use_transformers = use_transformers
        
        if use_transformers:
            # Use class-level singleton for model to avoid reloading
            if TextSummarizer._model_instance is None:
                logging.info("Loading Sentence Transformer model...")
                TextSummarizer._model_instance = SentenceTransformer('distilbert-base-multilingual-cased')
                logging.info("Model loaded successfully")
            self.model = TextSummarizer._model_instance
        
        # TF-IDF parameters
        self.tfidf_params = {
            'max_df': 0.95,
            'min_df': 2,
            'ngram_range': (1, 2),
            'strip_accents': None if language == 'turkish' else 'unicode'
        }
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(**self.tfidf_params)

    def _filter_sentences(self, sentences: List[str]) -> List[Tuple[str, int]]:
        """Filter out very short and very long sentences."""
        filtered = []
        for idx, sent in enumerate(sentences):
            words = len(word_tokenize(sent))
            if 3 <= words <= 50:  # Adjustable thresholds
                filtered.append((sent, idx))
            else:
                logging.debug(f"Filtered out sentence {idx}: Length {words} words")
        return filtered

    def _calculate_similarity(self, sentences: List[str]) -> np.ndarray:
        if self.use_transformers:
            # Use sentence transformers for better semantic understanding
            logging.info("Calculating sentence embeddings...")
            embeddings = self.model.encode(sentences, show_progress_bar=False)
            similarity_matrix = cosine_similarity(embeddings)
            logging.info("Similarity calculation complete")
        else:
            # Use TF-IDF with optimized parameters
            logging.info("Using TF-IDF for similarity calculation")
            try:
                X = self.vectorizer.fit_transform(sentences)
                similarity_matrix = cosine_similarity(X, X)
            except Exception as e:
                logging.error(f"Error in TF-IDF calculation: {e}")
                raise

        return similarity_matrix

    def summarize_text(self, text: str, n_sentences: int = 3, debug: bool = False) -> Dict:
        # Tokenize into sentences
        sentences = nltk.sent_tokenize(text)
        original_sentences = sentences.copy()
        
        if len(sentences) <= n_sentences:
            return {
                'summary': text,
                'scores': None,
                'debug_info': None
            }

        # Filter sentences by length and preserve original indices
        filtered_sentences = self._filter_sentences(sentences)
        if not filtered_sentences:
            return {
                'summary': text,
                'scores': None,
                'debug_info': "No sentences passed length filtering"
            }

        filtered_sents, original_indices = zip(*filtered_sentences)

        # Preprocess sentences for similarity calculation
        processed_sentences = [
            self.preprocessor.preprocess(sent)
            for sent in filtered_sents
        ]

        # Calculate similarity
        try:
            similarity_matrix = self._calculate_similarity(processed_sentences)
        except Exception as e:
            logging.error(f"Error in similarity calculation: {e}")
            return {
                'summary': text,
                'scores': None,
                'debug_info': f"Error in similarity calculation: {str(e)}"
            }

        # Create graph and calculate scores
        graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(graph)

        # Combine scores with original indices and sentences
        ranked_sentences = [
            (scores[i], filtered_sents[i], original_indices[i])
            for i in range(len(filtered_sents))
        ]

        # Sort by original position for final summary
        selected_sentences = sorted(
            sorted(ranked_sentences, key=lambda x: x[0], reverse=True)[:n_sentences],
            key=lambda x: x[2]
        )

        # Prepare debug information
        debug_info = {
            'sentence_scores': {
                sent: score for score, sent, _ in ranked_sentences
            },
            'preprocessing_info': {
                'original_length': len(sentences),
                'filtered_length': len(filtered_sents),
                'model_type': 'Sentence Transformers' if self.use_transformers else 'TF-IDF',
                'filtered_sentences': len(filtered_sents),
                'average_score': sum(scores.values()) / len(scores) if scores else 0
            }
        } if debug else None

        # Create summary
        summary = ' '.join(sent for _, sent, _ in selected_sentences)

        return {
            'summary': summary,
            'scores': {sent: score for score, sent, _ in selected_sentences},
            'debug_info': debug_info
        }
