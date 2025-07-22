import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

movies = pd.read_csv('/content/drive/MyDrive/[01] Data Engineer 2025/final/dataset/smal/movies.csv')
ratings = pd.read_csv('/content/drive/MyDrive/[01] Data Engineer 2025/final/dataset/smal/ratings.csv')

movies.head()

ratings.head()

df = pd.merge(ratings, movies, on='movieId')
df.drop_duplicates(inplace=True)
df.head()

"""#K-Means Genre Clustering"""

df['genres'] = df['genres'].str.split('|')
all_genres = set(genre for genres in df['genres'] for genre in genres)

for genre in all_genres:
    df[genre] = df['genres'].apply(lambda x: 1 if genre in x else 0)

df.drop(columns=['genres','(no genres listed)'], inplace=True)
df.head()

df['avg_rating'] = df.groupby('movieId')['rating'].transform('mean')
df['num_ratings'] = df.groupby('movieId')['rating'].transform('count')

df['year'] = df['title'].str.extract(r'\((\d{4})\)')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df = df.dropna(subset=['year'])

features_for_clustering = [
    'Western', 'Romance', 'Drama', 'Adventure', 'Horror', 'IMAX', 'Sci-Fi',
    'Children', 'Film-Noir', 'Animation', 'Action', 'Crime', 'Thriller', 'Comedy',
    'Musical', 'War', 'Documentary', 'Mystery', 'Fantasy','avg_rating', 'num_ratings',
]

print(features_for_clustering)

if all(col in df.columns for col in features_for_clustering):
    print("Semua kolom tersedia di DataFrame.")
else:
    print("Beberapa kolom tidak ditemukan di DataFrame.")

from sklearn.preprocessing import StandardScaler

X = df[features_for_clustering]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])

print(df.isnull().sum())

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, init='k-means++', n_init=30)
    kmeans.fit(X_pca)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", linestyle="-")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method untuk Menentukan K")
plt.show()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42, init='random')
df['cluster'] = kmeans.fit_predict(X_pca)
print(df['cluster'].value_counts())

print(df.groupby('cluster')[features_for_clustering].mean())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("Clustering Film Berdasarkan Genre")
plt.colorbar(label="Cluster")
plt.show()

from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score, calinski_harabasz_score

kmeans = KMeans(n_clusters=len(df['cluster'].unique()), random_state=42, n_init=10)
kmeans.fit(X_pca)
wcss = kmeans.inertia_

silhouette_avg = silhouette_score(X_pca, df['cluster'])
dbi = davies_bouldin_score(X_pca, df['cluster'])
chi = calinski_harabasz_score(X_pca, df['cluster'])

print(f"WCSS (Within-Cluster Sum of Squares): {wcss:.4f}")
print(f"Silhouette Score: {silhouette_avg:.4f}")
print(f"Davies-Bouldin Index: {dbi:.4f}")
print(f"Calinski-Harabasz Index: {chi:.4f}")

df.info()

df.head()

"""#NCF"""

df.info()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Mapping userId & movieId agar mulai dari 0
user_id_mapping = {id: idx for idx, id in enumerate(df['userId'].unique())}
movie_id_mapping = {id: idx for idx, id in enumerate(df['movieId'].unique())}

df['userId'] = df['userId'].map(user_id_mapping)
df['movieId'] = df['movieId'].map(movie_id_mapping)
df['cluster'] = df['cluster'].astype('category').cat.codes  # Cluster mulai dari 0

# 🔹 Split train & test
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# 🔹 Normalisasi avg_rating dan num_ratings
scaler = MinMaxScaler()
train_data[['avg_rating', 'num_ratings']] = scaler.fit_transform(train_data[['avg_rating', 'num_ratings']])
test_data[['avg_rating', 'num_ratings']] = scaler.transform(test_data[['avg_rating', 'num_ratings']])

# 🔹 Dataset untuk training
class MovieDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userId'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movieId'].values, dtype=torch.long)
        self.clusters = torch.tensor(df['cluster'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float32)
        self.avg_ratings = torch.tensor(df['avg_rating'].values, dtype=torch.float32)
        self.num_ratings = torch.tensor(df['num_ratings'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return (self.users[idx], self.movies[idx], self.clusters[idx],
                self.avg_ratings[idx], self.num_ratings[idx]), self.ratings[idx]

# 🔹 DataLoader
train_dataset = MovieDataset(train_data)
test_dataset = MovieDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

# 🔹 Model NCF dengan Dropout
class NCF(nn.Module):
    def __init__(self, num_users, num_movies, num_clusters, embedding_dim=64):
        super(NCF, self).__init__()

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.cluster_embedding = nn.Embedding(num_clusters, embedding_dim)

        # Fully Connected Layers dengan Dropout
        self.fc1 = nn.Linear(embedding_dim * 3 + 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Dropout untuk regularisasi

    def forward(self, user, movie, cluster, avg_rating, num_ratings):
        user_emb = self.user_embedding(user)
        movie_emb = self.movie_embedding(movie)
        cluster_emb = self.cluster_embedding(cluster)

        # Gabungkan semua fitur
        x = torch.cat([user_emb, movie_emb, cluster_emb, avg_rating.unsqueeze(1), num_ratings.unsqueeze(1)], dim=1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout di sini
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # Output ukuran (batch_size, 1)

        return x.squeeze(1)  # Pastikan output menjadi (batch_size,)

# 🔹 Inisialisasi Model
num_users = df['userId'].nunique()
num_movies = df['movieId'].nunique()
num_clusters = df['cluster'].nunique()

model = NCF(num_users, num_movies, num_clusters)

criterion = nn.MSELoss()  # Mean Squared Error cocok untuk rating prediksi
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Turunkan learning rate

# 🔹 Training Function dengan Early Stopping
def train_model(model, train_loader, criterion, optimizer, epochs=50, patience=5):
    model.train()
    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            (users, movies, clusters, avg_ratings, num_ratings), ratings = batch
            optimizer.zero_grad()
            outputs = model(users, movies, clusters, avg_ratings, num_ratings)
            loss = criterion(outputs, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered!")
                break

# 🔹 Jalankan Training
train_model(model, train_loader, criterion, optimizer, epochs=50)

# 🔹 Evaluasi Model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            (users, movies, clusters, avg_ratings, num_ratings), ratings = batch
            outputs = model(users, movies, clusters, avg_ratings, num_ratings)
            loss = criterion(outputs, ratings)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_loader):.4f}")

evaluate_model(model, test_loader, criterion)

# 🔹 Evaluasi RMSE & MAE
def rmse_loss(y_true, y_pred):
    return torch.sqrt(torch.mean((y_true - y_pred) ** 2))

def evaluate_rmse(model, test_loader):
    model.eval()
    total_rmse = 0
    with torch.no_grad():
        for batch in test_loader:
            (users, movies, clusters, avg_ratings, num_ratings), ratings = batch
            outputs = model(users, movies, clusters, avg_ratings, num_ratings)
            total_rmse += rmse_loss(ratings, outputs).item()
    print(f"Test RMSE: {total_rmse / len(test_loader):.4f}")

evaluate_rmse(model, test_loader)

def mae_loss(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def evaluate_mae(model, test_loader):
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for batch in test_loader:
            (users, movies, clusters, avg_ratings, num_ratings), ratings = batch
            outputs = model(users, movies, clusters, avg_ratings, num_ratings)
            total_mae += mae_loss(ratings, outputs).item()
    print(f"Test MAE: {total_mae / len(test_loader):.4f}")

evaluate_mae(model, test_loader)

import pandas as pd
import torch

movies_df = movies
ratings_df = ratings

ratings_summary = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
ratings_summary.columns = ["movieId", "avg_rating", "num_ratings"]
movies_info = pd.merge(movies_df, ratings_summary, on="movieId", how="left")

movie_details = movies_info.set_index("movieId").to_dict(orient="index")

def recommend_movies(model, user_id, df, movie_id_mapping, user_id_mapping, top_k=10):
    model.eval()

    mapped_user_id = user_id_mapping.get(user_id, None)
    if mapped_user_id is None:
        print("User ID tidak ditemukan.")
        return []

    all_movie_ids = set(movie_id_mapping.values())
    watched_movies = set(df[df['userId'] == mapped_user_id]['movieId'].values)
    unseen_movies = list(all_movie_ids - watched_movies)
    user_tensor = torch.tensor([mapped_user_id] * len(unseen_movies), dtype=torch.long)
    movie_tensor = torch.tensor(unseen_movies, dtype=torch.long)

    cluster_tensor = torch.tensor([0] * len(unseen_movies), dtype=torch.long)
    avg_rating_tensor = torch.tensor([df['avg_rating'].mean()] * len(unseen_movies), dtype=torch.float32)
    num_ratings_tensor = torch.tensor([df['num_ratings'].mean()] * len(unseen_movies), dtype=torch.float32)

    with torch.no_grad():
        predictions = model(user_tensor, movie_tensor, cluster_tensor, avg_rating_tensor, num_ratings_tensor)
    top_indices = torch.argsort(predictions, descending=True)[:top_k]
    recommended_movie_ids = [
        list(movie_id_mapping.keys())[list(movie_id_mapping.values()).index(movie)]
        for movie in torch.tensor(unseen_movies)[top_indices].tolist()
    ]
    return recommended_movie_ids

user_id = 2
recommended_movie_ids = recommend_movies(model, user_id, df, movie_id_mapping, user_id_mapping, top_k=10)
recommended_movies = [
    {
        "movieId": movie_id,
        "title": movie_details.get(movie_id, {}).get("title", "Unknown"),
        "genres": movie_details.get(movie_id, {}).get("genres", "Unknown"),
        "avg_rating": round(movie_details.get(movie_id, {}).get("avg_rating", 0), 2),
        "num_ratings": movie_details.get(movie_id, {}).get("num_ratings", 0),
    }
    for movie_id in recommended_movie_ids
]

recommended_movies_df = pd.DataFrame(recommended_movies)
recommended_movies_df.head()