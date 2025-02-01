import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.metrics.pairwise import cosine_similarity
import nltk 
from nltk.corpus import stopwords 
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('vader_lexicon')
#import spacy #for faster tokenization and lemmatization
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import string
from utils.helpers import preprocess_data


# Calculate Cosine Similarity between recipes
def calculate_similarity(pt):
    return cosine_similarity(pt)

def recommend(recipe_name, pt, similarity_scores):
    # Normalize the input recipe_name similarly
    recipe_name = recipe_name.strip().lower()
    
    if recipe_name not in pt.index:
        raise ValueError(f"Recipe '{recipe_name}' not found in the dataset. Please check the name or choose another recipe.")
    
    index = np.where(pt.index == recipe_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = []
    for i in similar_items:
        recommendations.append(pt.index[i[0]])
    
    return recommendations

# Main function to run the recommendation
def run_collaborative(recipe_name):
    pt = preprocess_data()
    similarity_scores = calculate_similarity(pt)
    recommendations = recommend(recipe_name, pt, similarity_scores)
    return recommendations