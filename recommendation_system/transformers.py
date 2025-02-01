import os
import pickle
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
# Uncomment the next line if you want to do PCA visualization
# from sklearn.decomposition import PCA
from utils.helpers import load_data  # Ensure load_data is implemented properly

# ------------------------------
# TransformerRecommender Class
# ------------------------------

class TransformerRecommender:
    def __init__(self, 
                 model_name='sentence-transformers/all-MiniLM-L6-v2', 
                 n_neighbors=10, 
                 cache_path='./embeddings/embeddings_cache.pkl'):
        """
        Initializes the transformer model, caches embeddings if available, and sets up a nearest neighbor model.
        """
        self.model_name = model_name
        self.n_neighbors = n_neighbors
        self.cache_path = cache_path

        # Initialize the transformer model
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.popularity_df = None
        self.nn = None

        # Load cached embeddings if available; otherwise compute and cache them.
        self._prepare()

    def _prepare(self):
        """
        Prepares the embeddings and associated data. Loads from cache if possible,
        else loads data, computes embeddings, caches the results, and fits the nearest neighbor model.
        """
        if os.path.exists(self.cache_path):
            st.write("Loading cached embeddings...")
            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            self.embeddings = cache_data.get('embeddings')
            self.popularity_df = cache_data.get('popularity_df')
        else:
            st.write("Cache not found. Loading data and computing embeddings...")
            # Load your data (adjust the load_data() function as needed)
            recipes, reviews, popularity_df = load_data()
            self.popularity_df = popularity_df

            # Extract the descriptions (making sure to drop null values)
            descriptions = popularity_df['Description'].dropna().tolist()
            # Compute embeddings using the transformer model
            self.embeddings = self.model.encode(descriptions, show_progress_bar=True)

            # Cache the computed embeddings and the data for future runs
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'embeddings': self.embeddings, 'popularity_df': self.popularity_df}, f)
            st.write("Embeddings computed and cached.")

        # Fit the nearest neighbor search model using the embeddings
        self.nn = NearestNeighbors(n_neighbors=self.n_neighbors)
        self.nn.fit(self.embeddings)

    def embed(self, texts):
        """
        Generate embeddings for a list of texts.
        """
        return self.model.encode(texts)

    def recommend(self, text):
        """
        Given an input text, returns recommendations by finding its nearest neighbors.
        """
        # Compute embedding for the query text
        query_emb = self.embed([text])
        # Find the nearest neighbors indices
        neighbor_indices = self.nn.kneighbors(query_emb, return_distance=False)[0]
        # Return a subset of the data (e.g., 'Name' and 'RecipeIngredientParts') corresponding to those indices
        return self.popularity_df[['Name', 'RecipeIngredientParts']].iloc[neighbor_indices]


