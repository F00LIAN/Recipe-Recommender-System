# app.py
import streamlit as st
from recommendation_system.collaborative import run_collaborative
from recommendation_system.transformers import TransformerRecommender
import matplotlib.pyplot as plt
# Uncomment and import PCA if needed:
# from sklearn.decomposition import PCA

def main():
    st.title("Food Recommendation System")

    # Let the user choose the recommendation method.
    method = st.radio("Select Recommendation Method", 
                        ["Collaborative Filtering", "Deep Learning Transformers"])

    # User input for a recipe name or description
    recipe_name = st.text_input("Enter a recipe name:")

    # For demonstration purposes, here's a simple collaborative filtering simulation.
    if method == "Collaborative Filtering":
        def run_collaborative(recipe_name):
            # Replace this with your actual collaborative filtering logic.
            return [f"Collaborative recommendation 1 for '{recipe_name}'",
                    f"Collaborative recommendation 2 for '{recipe_name}'"]

        if recipe_name:
            try:
                recommendations = run_collaborative(recipe_name)
                st.subheader(f"Collaborative Filtering Recommendations for '{recipe_name}':")
                for rec in recommendations:
                    st.write(rec)
            except ValueError as e:
                st.error(str(e))

    # Deep Learning Transformers-based recommendation
    elif method == "Deep Learning Transformers":
        if recipe_name:
            st.subheader(f"Transformer-based Recommendations for '{recipe_name}':")
            # Initialize the recommender (this will load cached embeddings if available)
            transformer_recommender = TransformerRecommender()
            # Get recommendations
            recommendations = transformer_recommender.recommend(recipe_name)
            st.write(recommendations)

            # Optionally, perform PCA and visualize the embedding space
            # Uncomment the following lines if you wish to see a 2D visualization.
            
            #pca = PCA(n_components=2)
            #emb_viz = pca.fit_transform(transformer_recommender.embeddings)
            #fig, ax = plt.subplots()
            #ax.scatter(emb_viz[:, 0], emb_viz[:, 1])
            #ax.set_title("Embedding Space")
            #st.pyplot(fig)
            
if __name__ == "__main__":
    main()
