import json
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the JSON files
def load_streaming_history(filepaths):
    data_frames = []
    for filepath in filepaths:
        with open(filepath, 'r') as file:
            data = json.load(file)
            data_frames.append(pd.DataFrame(data))
    return pd.concat(data_frames, ignore_index=True)

# Filepaths to the streaming history JSON files
filepaths = [
    '../data/StreamingHistory_music_0.json',
    '../data/StreamingHistory_music_1.json',
    '../data/StreamingHistory_music_2.json'
]

# Load data
df = load_streaming_history(filepaths)

# Data preprocessing
df['endTime'] = pd.to_datetime(df['endTime'])
df['artistName'] = df['artistName'].astype(str)
df['trackName'] = df['trackName'].astype(str)
df['msPlayed'] = pd.to_numeric(df['msPlayed'], errors='coerce')
df = df.dropna()

# Create a user-item interaction matrix
user_item_matrix = df.pivot_table(index='artistName', columns='trackName', values='msPlayed', aggfunc='sum', fill_value=0)

# Collaborative Filtering using K-Nearest Neighbors
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(user_item_matrix.T)  # Note: Transpose the matrix to fit correctly

def get_knn_recommendations(user_item_matrix, track_name, k=10):
    if track_name not in user_item_matrix.columns:
        return ["Track not found in the dataset"]
    distances, indices = model_knn.kneighbors(user_item_matrix.T.loc[track_name].values.reshape(1, -1), n_neighbors=k+1)
    recommended_tracks = [user_item_matrix.columns[i] for i in indices.flatten() if i < len(user_item_matrix.columns)][1:]  # Exclude the track itself
    return recommended_tracks

# Content-Based Filtering using TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['trackName'])

# Calculate cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_based_recommendations(track_name, cosine_sim=cosine_sim):
    if track_name not in df['trackName'].values:
        return ["Track not found in the dataset"]
    track_index = df[df['trackName'] == track_name].index[0]
    similarity_scores = list(enumerate(cosine_sim[track_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    recommended_tracks = [df['trackName'][i[0]] for i in similarity_scores[1:11]]
    return recommended_tracks

# Hybrid Recommendation System
def hybrid_recommendations(user_item_matrix, track_name, k=10):
    knn_recommendations = get_knn_recommendations(user_item_matrix, track_name, k)
    content_based_recommendations = get_content_based_recommendations(track_name)
    hybrid_recommendations = list(set(knn_recommendations + content_based_recommendations))
    return hybrid_recommendations[:k]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    track_name = request.form['track_name']
    knn_recommendations = get_knn_recommendations(user_item_matrix, track_name)
    content_based_recommendations = get_content_based_recommendations(track_name)
    hybrid_recommendations = hybrid_recommendations(user_item_matrix, track_name)
    
    return render_template('recommendations.html', 
                            track_name=track_name, 
                            knn_recommendations=knn_recommendations, 
                            content_based_recommendations=content_based_recommendations, 
                            hybrid_recommendations=hybrid_recommendations)

if __name__ == '__main__':
    app.run(debug=True)
