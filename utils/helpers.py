import pandas as pd
import numpy as np

def load_data():
    recipes = pd.read_parquet('./data/recipes.parquet')
    reviews = pd.read_parquet('./data/reviews.parquet')
    popularity = pd.read_parquet('./data/popular_par.parquet')
    return recipes, reviews, popularity

def preprocess_data():
    recipes, reviews, popularity = load_data()
    ratings_with_name = recipes.merge(reviews, on='RecipeId')

    # Find users that have rated over 200 recipes
    x = ratings_with_name.groupby('AuthorId_x').count()['Rating'] > 200
    trusted_users = x[x].index

    filtered_rating = ratings_with_name[ratings_with_name['AuthorId_x'].isin(trusted_users)]

    # Find recipes with most ratings by users
    y = filtered_rating.groupby('Name').count()['Rating'] >= 50
    famous_recipes = y[y].index

    # Create a DataFrame of the most famous recipes rated by the most trusted users
    final_ratings = filtered_rating[filtered_rating['Name'].isin(famous_recipes)]
    
    # Normalize recipe names: strip spaces and convert to lowercase
    final_ratings['Name'] = final_ratings['Name'].str.strip().str.lower()
    
    pt = final_ratings.pivot_table(index='Name', columns='AuthorId_x', values='Rating')
    pt.fillna(0, inplace=True)
    return pt