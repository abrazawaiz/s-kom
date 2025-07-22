# 📦 Install Library
!pip install scikit-learn tensorflow keras matplotlib seaborn

# 📂 Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# 📊 Fungsi untuk Visualisasi
def plot_history(history):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def visualize_clusters(X_pca, labels, centroids=None, title="Cluster Visualization"):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(scatter, label='Cluster')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

def plot_elbow_method(X, max_k=10):
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        if k > 1:  # Silhouette score needs at least 2 clusters
            labels = kmeans.labels_
            silhouette_scores.append(silhouette_score(X, labels))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, 'o-')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_range[0:], silhouette_scores, 'o-')
    plt.title('Silhouette Score Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# 📂 Load Dataset
print("Loading datasets...")
movies = pd.read_csv('/content/drive/MyDrive/final/data/movies.csv')
ratings = pd.read_csv('/content/drive/MyDrive/final/data/ratings.csv')

# 🧹 Data Cleaning
print("Cleaning data...")
# Menangani missing values jika ada
movies.dropna(subset=['movieId', 'title'], inplace=True)
ratings.dropna(inplace=True)

# Memastikan ratings dalam rentang yang valid (1-5)
ratings = ratings[(ratings['rating'] >= 1) & (ratings['rating'] <= 5)]

# 📊 EDA Awal
print("Performing initial EDA...")
print(f"Total movies: {len(movies)}")
print(f"Total ratings: {len(ratings)}")
print(f"Total users: {ratings['userId'].nunique()}")

# Distribusi rating
plt.figure(figsize=(10, 6))
sns.histplot(ratings['rating'], kde=True, bins=10)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

# 🔄 Feature Engineering
print("Performing feature engineering...")

# Film Year Extraction
movies['year'] = movies['title'].str.extract(r'\((\d{4})\)').astype('float')
movies['title_length'] = movies['title'].apply(len)

# One-Hot Encoding Genre dengan pendekatan yang lebih terstruktur
movies['genres'] = movies['genres'].str.split('|')
all_genres = set(genre for genres in movies['genres'] for genre in genres if genre != '(no genres listed)')

genre_df = pd.DataFrame()
for genre in all_genres:
    genre_df[genre] = movies['genres'].apply(lambda x: 1 if genre in x else 0)

# Gabungkan dengan dataframe movies
movies = pd.concat([movies, genre_df], axis=1)
movies.drop(columns=['genres'], inplace=True)

# Menambahkan fitur untuk rating
ratings_agg = ratings.groupby('movieId').agg({
    'rating': ['mean', 'std', 'count']
}).reset_index()
ratings_agg.columns = ['movieId', 'avg_rating', 'std_rating', 'rating_count']

# Menangani nilai NaN di std_rating
ratings_agg['std_rating'].fillna(0, inplace=True)

# Menggabungkan dengan dataframe movies
movies = movies.merge(ratings_agg, on='movieId', how='left')

# Mengisi nilai NaN di kolom yang baru ditambahkan
movies['avg_rating'].fillna(movies['avg_rating'].mean(), inplace=True)
movies['std_rating'].fillna(0, inplace=True)
movies['rating_count'].fillna(0, inplace=True)

# Menghitung popularitas dari rating_count
movies['popularity'] = np.log1p(movies['rating_count'])

# 🔄 Normalisasi Fitur
print("Normalizing features...")

# Memilih fitur untuk clustering
genre_cols = list(all_genres)
numeric_cols = ['year', 'avg_rating', 'std_rating', 'popularity']
feature_cols = genre_cols + numeric_cols

# Memisahkan fitur yang akan digunakan
X = movies[feature_cols].copy()
X.fillna(X.mean(), inplace=True)  # Menangani nilai yang hilang

# Normalisasi data menggunakan RobustScaler (lebih tahan terhadap outlier)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols)

# Uji visualisasi distribusi sebelum dan sesudah scaling untuk beberapa fitur
sample_cols = numeric_cols
plt.figure(figsize=(15, 10))
for i, col in enumerate(sample_cols):
    plt.subplot(2, len(sample_cols), i+1)
    sns.histplot(X[col], kde=True)
    plt.title(f'Before Scaling: {col}')
    
    plt.subplot(2, len(sample_cols), i+len(sample_cols)+1)
    sns.histplot(X_scaled_df[col], kde=True)
    plt.title(f'After Scaling: {col}')
plt.tight_layout()
plt.show()

# 🎯 PCA untuk Reduksi Dimensi
print("Performing PCA...")
# Tentukan variance yang ingin dipertahankan
pca = PCA(n_components=0.9)  # Mempertahankan 90% variance
X_pca_full = pca.fit_transform(X_scaled)
print(f"Number of components to retain 90% variance: {X_pca_full.shape[1]}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'r-')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Components')
plt.show()

# Untuk visualisasi, gunakan hanya 2 komponen
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

# 🤖 Menentukan jumlah cluster optimal menggunakan Elbow Method
print("Finding optimal number of clusters...")
plot_elbow_method(X_pca_full, max_k=10)

# 🤖 K-Means Clustering dengan jumlah cluster yang ditentukan
n_clusters = 4  # Berdasarkan hasil Elbow Method
print(f"Performing K-Means clustering with {n_clusters} clusters...")
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca_full)

# Menambahkan cluster ke dataframe movies
movies['cluster'] = cluster_labels

# Visualisasi hasil clustering
visualize_clusters(X_pca, cluster_labels, kmeans.cluster_centers_, title=f"Movie Clustering (k={n_clusters})")

# 📊 Analisis Cluster
print("Analyzing clusters...")
# Karakteristik cluster berdasarkan genre
cluster_analysis = movies.groupby('cluster').agg({
    **{genre: 'mean' for genre in genre_cols},
    'avg_rating': 'mean',
    'rating_count': 'mean',
    'year': 'mean'
}).reset_index()

# Heatmap untuk distribusi genre di setiap cluster
plt.figure(figsize=(18, 8))
sns.heatmap(cluster_analysis[genre_cols], annot=True, cmap='viridis', fmt='.2f')
plt.title('Genre Distribution Across Clusters')
plt.tight_layout()
plt.show()

# Statistik untuk setiap cluster
for cluster in range(n_clusters):
    cluster_movies = movies[movies['cluster'] == cluster]
    print(f"\nCluster {cluster} Statistics:")
    print(f"Number of movies: {len(cluster_movies)}")
    print(f"Average rating: {cluster_movies['avg_rating'].mean():.2f}")
    print(f"Average popularity: {np.expm1(cluster_movies['popularity'].mean()):.0f} ratings")
    print(f"Average year: {cluster_movies['year'].mean():.0f}")
    
    # Top genres untuk cluster ini
    genre_means = pd.Series({genre: cluster_movies[genre].mean() for genre in genre_cols})
    top_genres = genre_means.sort_values(ascending=False).head(5)
    print(f"Top genres: {', '.join([f'{g} ({v:.2f})' for g, v in top_genres.items()])}")
    
    # Sample film dari cluster
    print("Sample movies:")
    print(cluster_movies.sort_values('avg_rating', ascending=False)[['title', 'avg_rating']].head(3))

# 📊 Evaluasi Clustering
print("\nEvaluating clustering quality...")
silhouette = silhouette_score(X_pca_full, cluster_labels)
davies_bouldin = davies_bouldin_score(X_pca_full, cluster_labels)
calinski_harabasz = calinski_harabasz_score(X_pca_full, cluster_labels)

print(f"Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
print(f"Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
print(f"Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")

# 🔢 Gabungkan Data Ratings dan Movies dengan Clusters
print("Merging ratings with movies including clusters...")
df = ratings.merge(movies[['movieId', 'cluster']], on='movieId').drop_duplicates()

# Distribusi rating per cluster
plt.figure(figsize=(12, 6))
sns.boxplot(x='cluster', y='rating', data=df)
plt.title('Rating Distribution by Cluster')
plt.xlabel('Cluster')
plt.ylabel('Rating')
plt.show()

# 🔢 Label Encoding
print("Encoding categorical variables...")
user_encoder, movie_encoder = LabelEncoder(), LabelEncoder()
df['userId'] = user_encoder.fit_transform(df['userId'])
df['movieId'] = movie_encoder.fit_transform(df['movieId'])

# 🎯 Split Data Train & Test
print("Splitting data into training and test sets...")
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training set size: {len(train)}, Test set size: {len(test)}")

# 🎥 Model NCF (Neural Collaborative Filtering) dengan peningkatan
print("Building and training the NCF model...")
# Hyperparameters
embedding_dim = 64
dropout_rate = 0.3  

# Model architecture
user_input = Input(shape=(1,), name='user_input')
movie_input = Input(shape=(1,), name='movie_input')
cluster_input = Input(shape=(1,), name='cluster_input')

# Embeddings
user_embedding = Embedding(df['userId'].nunique(), embedding_dim, name='user_embedding')(user_input)
movie_embedding = Embedding(df['movieId'].nunique(), embedding_dim, name='movie_embedding')(movie_input)
cluster_embedding = Embedding(n_clusters, embedding_dim // 2, name='cluster_embedding')(cluster_input)

# Flatten embeddings
user_vec = Flatten(name='user_flatten')(user_embedding)
movie_vec = Flatten(name='movie_flatten')(movie_embedding)
cluster_vec = Flatten(name='cluster_flatten')(cluster_embedding)

# Combine features
concat = Concatenate(name='concat_features')([user_vec, movie_vec, cluster_vec])
dropout1 = Dropout(dropout_rate)(concat)

# Deep layers
dense1 = Dense(128, activation='relu', name='dense_1')(dropout1)
dropout2 = Dropout(dropout_rate)(dense1)
dense2 = Dense(64, activation='relu', name='dense_2')(dropout2)
dropout3 = Dropout(dropout_rate)(dense2)
dense3 = Dense(32, activation='relu', name='dense_3')(dropout3)

# Output layer
output = Dense(1, activation='linear', name='prediction')(dense3)

# Create and compile model
model = Model(inputs=[user_input, movie_input, cluster_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# Model summary
model.summary()

# Callbacks for better training
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=0.00001
)

# 🚀 Train Model
print("Training the model...")
train_inputs = [train['userId'], train['movieId'], train['cluster']]
train_labels = train['rating']
test_inputs = [test['userId'], test['movieId'], test['cluster']]
test_labels = test['rating']

history = model.fit(
    train_inputs, 
    train_labels, 
    epochs=15, 
    batch_size=64, 
    validation_data=(test_inputs, test_labels),
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# Save model
model.save("ncf_model_improved.h5")

# Visualize training history
plot_history(history)

# 🏆 Evaluasi Model
print("\nEvaluating the model...")
loss, mae = model.evaluate(test_inputs, test_labels, verbose=1)
print(f"Test Loss (MSE): {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Prediksi pada test set
y_pred = model.predict(test_inputs).flatten()
y_true = test_labels.values

# RMSE
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"Test RMSE: {rmse:.4f}")

# Plot prediksi vs aktual
plt.figure(figsize=(10, 6))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'r-')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title('Actual vs Predicted Ratings')
plt.show()

# Plot error distribution
errors = y_true - y_pred
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=20, alpha=0.6)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.title('Distribution of Prediction Errors')
plt.axvline(x=0, color='r', linestyle='--')
plt.show()

# 🎥 Fungsi Rekomendasi Film yang Ditingkatkan
def recommend_movies(user_id, movies_df, ratings_df, cluster_df, model, top_n=10, min_ratings=5):
    # Handle kasus user tidak ada
    if user_id not in user_encoder.classes_:
        print(f"User {user_id} not found. Recommending popular movies.")
        popular_movies = movies_df.sort_values('rating_count', ascending=False)
        return popular_movies.head(top_n)['title'].tolist()
    
    # Mendapatkan encoded user ID
    user_idx = user_encoder.transform([user_id])[0]
    
    # Mendapatkan film yang sudah ditonton user
    watched_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].values
    
    # Mendapatkan semua film yang belum ditonton
    unwatched_movies = movies_df[~movies_df['movieId'].isin(watched_movies)]
    
    # Filter film dengan minimal ratings
    unwatched_movies = unwatched_movies[unwatched_movies['rating_count'] >= min_ratings]
    
    if len(unwatched_movies) == 0:
        print(f"No unwatched movies found for user {user_id}.")
        return []
    
    # Encode movie IDs
    encoded_movies = movie_encoder.transform(unwatched_movies['movieId'])
    
    # Get cluster information for each movie
    movie_clusters = unwatched_movies['cluster'].values
    
    # Prepare arrays for prediction
    user_array = np.array([user_idx] * len(encoded_movies))
    movie_array = encoded_movies
    cluster_array = movie_clusters
    
    # Make predictions
    predictions = model.predict([user_array, movie_array, cluster_array], verbose=0).flatten()
    
    # Get top N recommendations
    unwatched_movies['predicted_rating'] = predictions
    recommendations = unwatched_movies.sort_values('predicted_rating', ascending=False).head(top_n)
    
    return recommendations[['title', 'predicted_rating']]

# Contoh penggunaan
print("\nDemo: Generating movie recommendations...")
sample_user = ratings['userId'].sample(1).iloc[0]
print(f"Recommendations for user {sample_user}:")
recommendations = recommend_movies(sample_user, movies, ratings, movies, model)
print(recommendations)

# 📊 Analisis Final & Insights
print("\nGenerating final insights...")
print("1. Cluster Analysis:")
for i in range(n_clusters):
    cluster_size = (movies['cluster'] == i).sum()
    print(f"   - Cluster {i}: {cluster_size} movies ({cluster_size/len(movies)*100:.1f}%)")

print("\n2. Model Performance Summary:")
print(f"   - MAE: {mae:.4f}")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - MSE: {loss:.4f}")

print("\n3. Clustering Evaluation:")
print(f"   - Silhouette Score: {silhouette:.4f}")
print(f"   - Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"   - Calinski-Harabasz Index: {calinski_harabasz:.4f}")

print("\nProject completed successfully!")