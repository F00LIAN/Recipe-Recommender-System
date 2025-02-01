<p align="center">
  
https://github.com/user-attachments/assets/4cbf0230-f51e-407e-8cbd-f35d1bd39711

</p>

<h1>Food Recommendation System</h1>
<p>
  This repository contains a Streamlit application for food recommendations using Natural Language Processing. The project explores two different approaches:
  <strong>Traditional Recommendation Systems</strong> (collaborative filtering and content-based filtering) and 
  <strong>Deep Learning Transformers</strong> (leveraging pre-trained transformer models to extract semantic embeddings). These methods are compared and integrated to provide users with more accurate and context-aware food dish recommendations.
</p>
<br />

<h2>Try it Now!</h2>

- ### [Food Recommendation System Website](https://your-streamlit-app-link-here)

<h2>Environments and Technologies Used</h2>

- Python 3.7+
- Streamlit
- Pandas
- Scikit-Learn
- Sentence-Transformers (HuggingFace)
- Visual Studio Code

<h2>Operating Systems Used</h2>

- Windows 11

<h2>List of Prerequisites</h2>

Before running this project, ensure you have:
- Python 3.7+ installed.
- Required Python libraries (see `requirements.txt` for details): Pandas, Scikit-Learn, Sentence-Transformers, Streamlit.
- The Food.com Kaggle Dataset: [Food.com Recipes and Reviews](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

<h2>Running the App</h2>

### 1. Overview
This project demonstrates how two distinct recommendation systems can be built and compared for food recommendations:

- **Traditional Recommendation Systems:**  
  Uses collaborative filtering and content-based filtering (via cosine similarity) to generate recommendations. One limitation is that users must input the exact dish title as present in the dataset.

- **Deep Learning Transformers Recommendation System:**  
  Uses pre-trained transformer models from the HuggingFace library to extract embeddings from recipe descriptions. These embeddings capture contextual information better, and they are cached to speed up user queries. PCA is optionally used to visualize the embedding space.

### 2. The Data
The data used in this project comes from the Food.com Kaggle Dataset, which includes:
- **User Reviews:** Detailed feedback on recipes.
- **Food Dish Labels and Ingredients:** Information about various dishes and their components.

Data Link: [Food.com Recipes and Reviews on Kaggle](https://www.kaggle.com/datasets/irkaal/foodcom-recipes-and-reviews)

### 3. Traditional Recommendation System
In this approach:
- **Collaborative Filtering:** Recommends dishes based on user interactions and similarities.
- **Content-Based Filtering:** Utilizes cosine similarity on dish features (ingredients, labels) to compute similarity scores.
  
A limitation is that the user must input the dish name exactly as it appears in the dataset, which may not fully capture subtle relationships between dishes.

### 4. Deep Learning Transformers Recommendation System
This modern approach leverages transformer models to capture the semantic context of recipe descriptions:
- **Embedding Generation:** Uses pre-trained models (e.g., `sentence-transformers/all-MiniLM-L6-v2`) to generate embeddings for each recipe description.
- **Caching:** Precomputed embeddings are stored (using a caching mechanism) so that they need not be re-computed on every query, thus speeding up the recommendation process.
- **Visualization:** PCA is used to project high-dimensional embeddings into 2D space, providing a visual representation of how similar recipes are clustered.

Example code snippet for generating embeddings:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(recipe_descriptions, show_progress_bar=True)
