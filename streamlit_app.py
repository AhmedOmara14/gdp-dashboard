# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

# Step 2: Create Dummy Data
# User-Item Interaction Matrix with some missing ratings (represented as 0)
data = {
    'Item1': [5, 4, 1, 0, 0],
    'Item2': [4, 0, 0, 2, 4],
    'Item3': [1, 2, 3, 0, 0],
    'Item4': [0, 0, 4, 5, 3],
    'Item5': [0, 3, 0, 4, 0]
}

# Convert to DataFrame for visualization
df = pd.DataFrame(data, index=['User1', 'User2', 'User3', 'User4', 'User5'])
print("User-Item Interaction Matrix:")
print(df)

# Step 3: Apply Collaborative Filtering with SVD
# Convert the DataFrame to a NumPy matrix
interaction_matrix = df.to_numpy()

# Define the number of latent factors (components)
n_components = 2  # Adjust based on the complexity of the data

# Initialize and fit TruncatedSVD
svd = TruncatedSVD(n_components=n_components)
user_factors = svd.fit_transform(interaction_matrix)    # User factors
item_factors = svd.components_.T                        # Item factors

print("\nUser Factors Matrix (Reduced User-Item Matrix):")
print(user_factors)

print("\nItem Factors Matrix (Reduced Item-Feature Matrix):")
print(item_factors)

# Step 4: Predict Missing Ratings
# Reconstruct the matrix with predicted ratings
predicted_ratings = np.dot(user_factors, item_factors)

# Convert the predictions back to a DataFrame for easier viewing
predicted_df = pd.DataFrame(predicted_ratings, index=df.index, columns=df.columns)
print("\nPredicted User-Item Interaction Matrix (with predicted ratings):")
print(predicted_df)

# Step 5: Recommend Items for a User
def recommend_items(user, original_df, predicted_df, n_recommendations=2):
    # Find items not yet rated by the user in the original interaction matrix
    unrated_items = original_df.loc[user][original_df.loc[user] == 0].index
    
    # Get the predicted ratings for the unrated items
    recommendations = predicted_df.loc[user, unrated_items].sort_values(ascending=False)
    
    # Return the top-N recommendations
    return recommendations.head(n_recommendations)

# Example: Recommend top 2 items for User1
user = 'User1'
recommendations = recommend_items(user, df, predicted_df)
print(f"\nTop recommendations for {user}:")
print(recommendations)
