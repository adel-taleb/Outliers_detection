import os
import joblib
import pandas as pd
import torch
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors
from cuml.manifold import TSNE
from sklearn.covariance import EmpiricalCovariance
from scipy.spatial.distance import mahalanobis

class OutliersDetection:
    """
    Class for detecting outliers in a dataset using various outlier detection models.
    Provides methods for loading data, preprocessing, training models, evaluating, and saving results.
    """
    
    def __init__(self, model_conf, model_name, sub_dataset, add_numerical_data, save_dir):
        """
        Initialize the outliers detection class with the given configuration.

        Parameters:
        - model_conf: Configuration of the outlier detection model
        - model_name: Name of the model (e.g., 'Isolation Forest')
        - sub_dataset: The Amazon Reviews sub-dataset to load
        - add_numerical_data: Boolean to add numerical features
        - save_dir: Directory to save models and evaluation metrics
        """
        print("Torch availability:", torch.cuda.is_available()) 
        print("Num GPUs Available for TensorFlow:", len(tf.config.list_physical_devices('GPU')))
        
        # Initialize parameters
        self.model_name = model_name
        self.sub_dataset = sub_dataset
        self.add_numerical_data = add_numerical_data
        self.model_conf = model_conf
        self.save_dir = save_dir
        
        # Load, preprocess data, train and evaluate
        self.reviews_dataset, self.meta_dataset = self.load_data(sub_dataset)
        self.X_train, self.X_test = self.preprocess(add_numerical_data)
        self.model, self.y_train_labels, self.outlier_scores = self.train(self.X_train, model_name, model_conf)
        
        # Set evaluation options and evaluate model
        self.silhouette = True
        self.davies_bouldin = True
        self.nearest_neighbor = True
        self.eval(model_name, save_dir=self.save_dir)
        
        # Save the trained model
        self.save_model(model_name, self.model, save_dir)
        self.tsne_compute(self.X_train, save_dir, filename="tsne_plot_train.png")
        self.tsne_compute(self.X_test, save_dir, filename="tsne_plot_test.png")
        self.distribution_shift_scoring(self.X_train, self.X_test, save_dir=save_dir)
        
    def load_data(self, sub_dataset):
        """
        Load the Amazon reviews and metadata for the specified sub-dataset.

        Parameters:
        - sub_dataset: The Amazon sub-dataset to load (default: "Office_Products")

        Returns:
        - reviews_dataset: Loaded dataset of product reviews
        - meta_dataset: Loaded dataset of product metadata
        """
        # Load product reviews
        reviews_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{sub_dataset}",
            split="full",
            trust_remote_code=True
        )
        
        # Load product metadata
        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{sub_dataset}",
            split="full",
            trust_remote_code=True
        )
        return reviews_dataset, meta_dataset
        
    def preprocess(self, add_numerical_data=False):
        """
        Preprocess the datasets by merging metadata with reviews, embedding text, and optionally adding numerical features.

        Parameters:
        - add_numerical_data: Whether to add additional numerical features

        Returns:
        - X_train, X_test: Train and test sets after preprocessing
        """
        # Convert datasets to pandas DataFrames and merge
        meta_df = self.meta_dataset.to_pandas()
        reviews_df = self.reviews_dataset.to_pandas()
        merged_df = pd.merge(meta_df, reviews_df, on='parent_asin', how='left')
        
        # Standardize numerical features
        numerical_features = ['rating', 'helpful_vote']
        scaler = StandardScaler()
        merged_df[numerical_features] = scaler.fit_transform(merged_df[numerical_features])
        # Ensure text columns have no missing values by filling NaNs
        merged_df['title_x'] = merged_df['title_x'].fillna('')
        merged_df['title_y'] = merged_df['title_y'].fillna('')
        merged_df['text'] = merged_df['text'].fillna('')
        
        # Combine text data into a single column for embedding
        merged_df['textual_data'] = (
            'product title: ' + merged_df['title_x'].astype(str) + 
            '. review title: ' + merged_df['title_y'].astype(str) + 
            '. review text: ' + merged_df['text'].astype(str)
        )
        merged_df = merged_df.drop(columns=['title_x', 'title_y', 'text'])
        
        # Generate text embeddings
        text_embeddings = self._text_embedding(merged_df['textual_data'].tolist())
        embeddings_df = pd.DataFrame(text_embeddings)
        
        # Optionally add numerical data
        if add_numerical_data:
            other_features = merged_df[numerical_features].reset_index(drop=True)
            combined_df = pd.concat([embeddings_df, other_features], axis=1)
        else:
            combined_df = embeddings_df
        
        # Split data into train and test sets
        return train_test_split(combined_df.values, test_size=0.2, random_state=42)

    def _text_embedding(self, text_data, model_name='all-MiniLM-L6-v2'):
        """
        Embed text data using the specified model.

        Parameters:
        - text_data: List of text entries to embed
        - model: Name of the embedding model (default: 'all-MiniLM-L6-v2')

        Returns:
        - Embeddings for the provided text data
        """
        embedding_model = SentenceTransformer(model_name, device='cuda')
        return embedding_model.encode(text_data, show_progress_bar=True)
        
    def train(self, X_train, model_name, model):
        """
        Train the specified outlier detection model on the training data.

        Parameters:
        - X_train: Training data
        - model_name: Name of the model
        - model: Outlier detection model configuration

        Returns:
        - model: Trained model
        - y_train_labels: Outlier labels for training data
        - outlier_scores: Outlier scores for training data
        """
        print(f"\nTraining {model_name}...")
        model.fit(X_train)
        
        # Obtain outlier labels and scores
        y_train_labels = model.labels_  # Labels assigned by model
        outlier_scores = model.decision_scores_  # Outlier scores
        return model, y_train_labels, outlier_scores
        
    def eval(self, model_name, save_dir="saved_model"):
        """
        Evaluate the trained model and save evaluation metrics to a file.

        Parameters:
        - model_name: Name of the model
        - save_dir: Directory to save evaluation metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "model_evaluation_metrics.txt"), "w") as file:
            if self.silhouette:
                # Silhouette Score
                silhouette_avg = silhouette_score(self.X_train, self.y_train_labels)
                file.write(f"{model_name} - Silhouette Score: {silhouette_avg:.2f}\n")
                
            if self.davies_bouldin:
                # Davies-Bouldin Index
                db_index = davies_bouldin_score(self.X_train, self.y_train_labels)
                file.write(f"{model_name} - Davies-Bouldin Index: {db_index:.2f}\n")
                
            if self.nearest_neighbor:
                # Mean Distance to k-Nearest Neighbors
                nbrs = NearestNeighbors(n_neighbors=5).fit(self.X_train)
                distances, _ = nbrs.kneighbors(self.X_train)
                mean_distance = np.mean(distances, axis=1)
                file.write(f"{model_name} - Mean Distance to 5 Nearest Neighbors: {np.mean(mean_distance):.2f}\n")
            
            # Save a plot of outlier score distribution
            plt.figure()
            plt.hist(self.outlier_scores, bins=50)
            plt.xlabel("Outlier Score")
            plt.ylabel("Frequency")
            plt.title(f"{model_name} - Distribution of Outlier Scores")
            plt.savefig(os.path.join(save_dir, f"{model_name}_outlier_score_distribution.png"))
            plt.close()

    def save_model(self, model_name, model, save_dir):
        """
        Save the trained model to the specified directory.

        Parameters:
        - model_name: Name of the model
        - model: Trained model to save
        - save_dir: Directory to save the model
        """
        os.makedirs(save_dir, exist_ok=True)
        joblib.dump(model, os.path.join(save_dir, f"{model_name}.joblib"))
        print(f"Model saved in {save_dir} directory.")


    def tsne_compute(self, X, save_dir, filename="tsne_plot.png"):
        """
        Compute and visualize t-SNE on the data, saving the plot to a file.

        Parameters:
        - X: Data to visualize using t-SNE
        - labels: Optional labels for coloring the points (default: None)
        - save_dir: Directory to save the plot (default: 'saved_plots')
        - filename: Filename for the saved plot (default: 'tsne_plot.png')
        """
        labels = self.model.predict(X)
        # Configure t-SNE parameters optimized for large data on GPU
        tsne = TSNE(
            n_components=2,         # 2D output for visualization
            perplexity=50,          # Good starting value for large datasets
            learning_rate=500,      # Higher learning rate for larger data
            n_iter=1000,            # Start with 1000 iterations, increase if necessary
            early_exaggeration=12,  # Standard value
            init="random",          # Faster initialization
            verbose=True            # Output progress for monitoring
        )
        
        # Fit and transform your data using cuML t-SNE
        X_tsne = tsne.fit_transform(X)
        print("t-SNE output shape:", X_tsne.shape)
        
        # Create directory for saving the plot if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Plotting
        plt.figure(figsize=(10, 8))

        plt.scatter(X_tsne[labels == 0, 0], X_tsne[labels == 0, 1], label='Inliers', alpha=0.6)
        plt.scatter(X_tsne[labels == 1, 0], X_tsne[labels == 1, 1], label='Outliers', color='red', alpha=0.6)
        plt.title("t-SNE Visualization")
        plt.xlabel("t-SNE Component 1")
        plt.ylabel("t-SNE Component 2")
        plt.legend()
        # Save the plot
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        
        print(f"t-SNE plot saved at {save_path}")

        
    def distribution_shift_scoring(self, X_train, X_test, save_dir, filename="distribution_shift.png"):
        """
        Calculate and visualize the distribution shift scoring.

        Parameters:
        - X_train: Training data for calculating the baseline distribution
        - X_test: Test data to score for distribution shift
        - save_dir: Directory to save the plot (default: 'saved_plots')
        - filename: Filename for the saved plot (default: 'distribution_shift.png')
        """
        # Fit an Empirical Covariance model on the training data
        cov_model = EmpiricalCovariance().fit(X_train)
        train_scores = self._compute_mahalanobis_scores(X_train, cov_model)
        test_scores = self._compute_mahalanobis_scores(X_test, cov_model)
        
        # Save and display results
        self._visualize_distribution_shift(train_scores, test_scores, save_dir, filename)

    def _compute_mahalanobis_scores(self, X, cov_model):
        """
        Compute Mahalanobis distance scores as a measure of distribution shift.

        Parameters:
        - X: Data to calculate scores for
        - cov_model: Pre-trained covariance model for Mahalanobis distance

        Returns:
        - scores: List of Mahalanobis distance scores for each sample
        """
        mean_vec = cov_model.location_  # Mean vector from the covariance model
        cov_inv = np.linalg.inv(cov_model.covariance_)  # Inverse covariance matrix
        
        # Calculate Mahalanobis distance for each sample
        scores = [mahalanobis(x, mean_vec, cov_inv) for x in X]
        return np.array(scores)

    def _visualize_distribution_shift(self, train_scores, test_scores, save_dir, filename):
        """
        Visualize distribution shift by plotting score distributions for training and test data.

        Parameters:
        - train_scores: Mahalanobis scores for training data
        - test_scores: Mahalanobis scores for test data
        - save_dir: Directory to save the plot
        - filename: Filename for the saved plot
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot the score distributions
        plt.figure(figsize=(10, 6))
        plt.hist(train_scores, bins=30, alpha=0.6, label='Train Scores', color='blue')
        plt.hist(test_scores, bins=30, alpha=0.6, label='Test Scores', color='red')
        plt.xlabel("Distribution Shift Score (Mahalanobis Distance)")
        plt.ylabel("Frequency")
        plt.title("Distribution Shift Scoring for Training and Test Data")
        plt.legend()
        
        # Save plot
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path)
        plt.close()
        
        # Output basic score statistics
        print(f"Train set - Mean Score: {np.mean(train_scores):.2f}, Std: {np.std(train_scores):.2f}")
        print(f"Test set - Mean Score: {np.mean(test_scores):.2f}, Std: {np.std(test_scores):.2f}")
        print(f"Distribution shift plot saved at {save_path}")