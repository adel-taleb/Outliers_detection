import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

class OutliersTester:
    def __init__(self, model_path, embedding_model='all-MiniLM-L6-v2', device='cuda'):
        """
        Initialize the tester with the pre-trained model and embedding model.

        Parameters:
        - model_path: Path to the saved outlier detection model.
        - embedding_model: Name of the embedding model used (default: 'all-MiniLM-L6-v2').
        - device: Device to use for embedding model (default: 'cuda').
        """
        # Load the trained model
        self.model = joblib.load(model_path)
        # Initialize the embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device=device)

    def preprocess_input(self, product_title, review_title, review_text):
        """
        Preprocesses the input data by creating embeddings for the textual information.

        Parameters:
        - product_title: Title of the product.
        - review_title: Title of the review.
        - review_text: Text of the review.

        Returns:
        - embeddings: Embeddings for the combined input text.
        """
        # Combine the input text
        combined_text = f"product title: {product_title}. review title: {review_title}. review text: {review_text}"
        
        # Generate embeddings
        embeddings = self.embedding_model.encode([combined_text], show_progress_bar=False)
        return embeddings

    def predict_outlier(self, product_title, review_title, review_text):
        """
        Predicts if the review is an outlier.

        Parameters:
        - product_title: Title of the product.
        - review_title: Title of the review.
        - review_text: Text of the review.

        Returns:
        - is_outlier: Boolean indicating if the review is an outlier.
        """
        # Preprocess the input
        embeddings = self.preprocess_input(product_title, review_title, review_text)
        # Predict outlier label (1 = outlier, 0 = inlier)
        prediction = self.model.predict(embeddings)
        is_outlier = bool(prediction[0] == 1)
        return is_outlier
