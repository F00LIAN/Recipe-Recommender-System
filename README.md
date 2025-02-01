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

- ### [Food Recommendation System Website](https://deep-learning-recipe-recommendations.streamlit.app/)

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
```

### 5. Project Structure
```bash
.
├── README.md               <-- You are here
├── requirements.txt        <-- Python dependencies
├── app.py (or main.py)     <-- Primary Streamlit app script
├── utils/
│   └── helpers.py          <-- Data loading and utility functions
└── recommendation_system/
    ├── collaborative.py    <-- Traditional recommendation system code
    └── transformers.py     <-- Deep Learning Transformers recommendation system code
```

### 6. Dependencies and Installation
```bash
pip install -r requirements.txt
```

Dependencies include:

- Streamlit: For the interactive web app.
- Pandas: For data manipulation.
- Scikit-Learn: For similarity computations and other utilities.
- Sentence-Transformers: For generating contextual embeddings.
- Other standard libraries: (e.g., NumPy)

### 7. How the Code Works

```python
from utils.helpers import load_data
recipes, reviews, popularity_df = load_data()
```

- Loads recipe data, user reviews, and popularity metrics from the Food.com dataset.

**Traditional Methods**

- Uses cosine similarity to measure distances between dishes based on their features.

- Generates recommendations by comparing input dish titles with the dataset.

**Transformer-Based Methods**

- Computes embeddings for recipe descriptions using a pre-trained SentenceTransformer.

- Caches embeddings to avoid re-computation and speed up query processing.

- Uses a Nearest Neighbors algorithm to identify and recommend similar dishes.

**User Interface**

- The Streamlit app lets users choose between traditional and transformer-based recommendation systems.

- Users input a recipe name to receive relevant recommendations along with optional visualizations of the embedding space.

### 8. Next Steps

Future improvements for this project include:

  - Chatbot Integration: Enhance user experience by integrating a conversational interface.
  - Advanced UI/UX: Further refine the Streamlit interface for better usability.
  - Expanded Data Sources: Incorporate additional data (e.g., nutritional information, user preferences) for richer recommendations.
  - Model Enhancements: Experiment with newer deep learning architectures and fine-tuning techniques to improve recommendation accuracy.

<h2> Thank You! </h2> 

<p> Feel free to contribute, provide feedback, or reach out if you have any suggestions for improving this food recommendation system. If you found this project useful, please consider starring the repository and following for more insights into NLP and recommendation systems! </p> 

<p> Disclaimer: This project is for educational purposes only. The recommendations generated by this system should be used as a reference and not as definitive advice. Always perform additional research and consider multiple factors before making decisions. </p> 





