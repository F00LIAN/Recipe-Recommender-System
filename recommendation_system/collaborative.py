import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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
    similarity_scores = calculate_similarity(pt)
    recommendations = recommend(recipe_name, pt, similarity_scores)
    return recommendations
