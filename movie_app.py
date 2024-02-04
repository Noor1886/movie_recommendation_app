#!/usr/bin/env python
# coding: utf-8
import pandas as pd


# In[9]:


movies=pd.read_csv('movies.csv')

movies.head()


# In[11]:


import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np

# Function to split data evenly based on user IDs
def split_data_evenly(data, test_size=0.2):
    test_data = pd.DataFrame()
    train_data = pd.DataFrame()

    # Group data by user ID
    grouped = data.groupby('userId')

    for user_id, group in grouped:
        # Split each user's reviews evenly
        n = len(group)
        test_indices = np.random.choice(group.index, size=int(test_size * n), replace=False)

        # Assign reviews to test dataset
        test_data = pd.concat([test_data, group.loc[test_indices]])

        # Assign remaining reviews to training dataset
        train_data = pd.concat([train_data, group.drop(test_indices)])

    return train_data, test_data

# Load data
movies=pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')

# Merge ratings with movie titles
merged_data = pd.merge(ratings, movies, on='movieId')

# Convert genres into numerical features using one-hot encoding
genres = merged_data['genres'].str.split('|')
mlb = MultiLabelBinarizer()
genre_features = pd.DataFrame(mlb.fit_transform(genres), columns=mlb.classes_, index=merged_data.index)

# Concatenate genre features with ratings data
merged_data = pd.concat([merged_data, genre_features], axis=1)

# Split the data into training and testing sets
train_data, test_data = split_data_evenly(merged_data)

# Check GPU availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define PyTorch dataset
class RecommendationDataset(Dataset):
    def __init__(self, data):
        self.user_ids = torch.tensor(data['userId'].values, dtype=torch.long).to(device)
        self.movie_ids = torch.tensor(data['movieId'].values, dtype=torch.long).to(device)
        self.genre_features = torch.tensor(data.iloc[:, 6:].values, dtype=torch.float32).to(device)
        self.ratings = torch.tensor(data['rating'].values, dtype=torch.float32).to(device)

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.movie_ids[idx], self.genre_features[idx], self.ratings[idx]

# Create datasets and data loaders
train_dataset = RecommendationDataset(train_data)
test_dataset = RecommendationDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define model
class RecommendationModel(nn.Module):
    def __init__(self, num_users, num_movies, num_genres):
        super(RecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 50).to(device)
        self.movie_embedding = nn.Embedding(num_movies, 50).to(device)
        self.genre_linear = nn.Linear(num_genres, 50).to(device)
        self.concat_layer = nn.Linear(150, 128).to(device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization
        self.output_layer = nn.Linear(128, 1).to(device)
        self.sigmoid = nn.Sigmoid().to(device)


    def forward(self, user_ids, movie_ids, genre_features):
        user_embedded = self.user_embedding(user_ids)
        movie_embedded = self.movie_embedding(movie_ids)
        genre_linear = self.relu(self.genre_linear(genre_features))
        concatenated = torch.cat((user_embedded, movie_embedded, genre_linear), dim=1)
        out = self.relu(self.concat_layer(concatenated))
        out = self.output_layer(out)
        out = self.sigmoid(out)  # Apply sigmoid activation function
        out = out * 5.0
        return out.squeeze()

# Instantiate model and define loss function and optimizer
model = RecommendationModel(ratings['userId'].nunique()+1, movies['movieId'].max()+1, len(mlb.classes_))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# Training loop
for epoch in range(19):
    model.train()
    running_loss = 0.0
    for user_ids, movie_ids, genre_features, ratings in train_loader:

        optimizer.zero_grad()
        outputs = model(user_ids, movie_ids, genre_features)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}")

# Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for user_ids, movie_ids, genre_features, ratings in test_loader:
        outputs = model(user_ids, movie_ids, genre_features)
        test_loss += criterion(outputs, ratings).item()
print(f"Test Loss: {test_loss / len(test_loader)}")


# In[12]:


def get_top5_recommendations(model, user_Id, movie_Ids, genre_features):
    # Prepare input tensors
    user_Ids = torch.tensor(np.array([user_Id] * len(movie_Ids)),dtype=torch.long).to(device)
    movie_Ids = torch.tensor(movie_Ids,dtype=torch.long).to(device)
    genre_features = torch.tensor(genre_features.values,dtype=torch.float32).to(device)  # Assuming genre_features is a tensor


    print("t")

    # Forward pass to get predictions
    model.eval()
    with torch.no_grad():
        predictions = model(user_Ids, movie_Ids, genre_features).squeeze()

    print("h")

    # Sort predicted ratings in descending order and get top 5 indices
    top5_indices = predictions.argsort(descending=True)[:5]

    # Get top 5 movie IDs and predicted ratings
    top5_movie_ids = movie_Ids[top5_indices]
    top5_ratings = predictions[top5_indices]

    return top5_movie_ids, top5_ratings




# In[13]:


# Extract movie IDs from the preprocessed dataset
movie_ids = movies['movieId']

# Extract genre features from the preprocessed dataset # Assuming genre features start from the 7th column

# Print shapes of movie IDs and genre features
print("Shape of movie IDs:", movie_ids.shape)
print(type(movie_ids))

genre_features=movies['genres'].str.split('|')

genre_features = pd.DataFrame(mlb.fit_transform(genre_features), columns=mlb.classes_, index=movies.index)
print("Shape of genre features:", genre_features.shape)
print(type(genre_features))


# Example usage
user_id = 1  # User ID for which you want recommendations
# Assuming movie_ids and genre_features are available
top5_movie_ids, top5_ratings = get_top5_recommendations(model, user_id, movie_ids, genre_features)

# Print top 5 recommendations
print("Top 5 Recommendations for User", user_id)
for movie_id, rating in zip(top5_movie_ids, top5_ratings):
    print("Movie ID:", movie_id.item(), "Rating:", rating.item())


# In[14]:


import pickle


# In[15]:


with open("movie_app.pkl", 'wb') as file:
    pickle.dump(model, file)
    
print("Model saved successfully to movie_app")
    
    

