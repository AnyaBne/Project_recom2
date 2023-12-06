import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv('song_dataset.csv')  # Replace with the path to your dataset

# Configuration for collaborative filtering
reader = Reader(rating_scale=(df['play_count'].min(), df['play_count'].max()))
data = Dataset.load_from_df(df[['user', 'song', 'play_count']], reader)
algo = SVD()
trainset, _ = train_test_split(data, test_size=0.25)
algo.fit(trainset)

# Configuration for content-based filtering
df['combined_attributes'] = df['title'] + ' ' + df['release'] + ' ' + df['artist_name'] + ' ' + df['year'].astype(str)
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined_attributes'])

def explain_recommendations(user_id, n=10):
    # Obtenir les recommandations initiales
    recommended_songs = get_initial_recommendations(user_id, n)

    # Récupérer les scores de prédiction pour les chansons recommandées
    scores = []
    song_titles = recommended_songs['title'].tolist()  # Liste des titres de chansons recommandées

    for title in song_titles:
        # Trouver l'ID de la chanson correspondant au titre (si nécessaire)
        song_id = df[df['title'] == title]['song'].iloc[0]
        score = algo.predict(user_id, song_id).est
        scores.append(score)

    # Création du graphique à barres
    plt.figure(figsize=(10, 6))
    plt.bar(song_titles, scores, color='skyblue')
    plt.xlabel('Chansons')
    plt.ylabel('Scores de Prédiction')
    plt.title('Explication des Recommandations : Scores de Prédiction pour les Chansons Recommandées')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Affichage dans Streamlit
    st.pyplot(plt)

def explain_content_based_selection(selected_songs, recommended_songs_df, tfidf_vectorizer):
    explanations = []
    df['combined_attributes2'] = 'title: ' + df['title'] + ' ' + 'release: ' + df['release'] + ' ' + 'artist name: ' + df['artist_name'] + ' ' + 'year: ' + df['year'].astype(str)

    for song in selected_songs:
        selected_song_attributes = df[df['title'] == song]['combined_attributes2'].iloc[0]
        explanation = f"Selected song '{song}' has these key attributes: {selected_song_attributes}."
        explanations.append(explanation)

    explanations.append("\nBased on these attributes, the following songs are recommended:")

    for idx, rec_song in recommended_songs_df.iterrows():
        rec_song_title = rec_song['title']
        rec_song_attributes = df[df['title'] == rec_song_title]['combined_attributes'].iloc[0]
        explanation = f"Recommended song '{rec_song_title}' has similar attributes: {rec_song_attributes}."
        explanations.append(explanation)

    return explanations

# Streamlit app
def main():
    st.title("Song Recommendation System")

    # Step 1: User Login
    user_id = st.text_input("Enter your user ID:")
    if not user_id:
        st.warning("Please enter a user ID.")
        st.stop()

    # Step 2: Generate Initial Recommendations
    initial_recommendations = get_initial_recommendations(user_id)
    st.subheader("Initial Recommendations")
    st.write(initial_recommendations)

    if st.button('Explanation of this initial recommendation):
      explain_recommendations(user_id)

    # Step 3: User Refines Recommendations
    refined_recommendations = get_refined_recommendations(user_id)
    st.subheader("Refined Recommendations")
    st.write(refined_recommendations)
    

    
    if st.button('Explanation of refine recommendation:):
      explanations = explain_content_based_selection(selected_songs, final_recommendations_df, tfidf_vectorizer)        
      for explanation in explanations:
      st.info(explanation)

    # Step 5: Final Output
    final_recommendations = get_final_recommendations(user_id, initial_recommendations, refined_recommendations)
    st.subheader("Final Recommendations")
    st.write(final_recommendations)

# Helper functions
def get_initial_recommendations(user_id, n=10):
    # Collaborative filtering logic
    listened_songs = df[df['user'] == user_id]['song'].unique()
    all_songs = df[~df['song'].isin(listened_songs)]['song'].unique()

    predictions = []
    for song in all_songs:
        predictions.append((song, algo.predict(user_id, song).est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    recommended_song_ids = [song for song, _ in predictions[:n]]
    recommended_songs = df[df['song'].isin(recommended_song_ids)][['song', 'title']].drop_duplicates().head(n)

    return recommended_songs

def get_refined_recommendations(user_id):
    # Implement logic to allow the user to refine recommendations
    # Return a list of user-refined recommendations
    # For example, you can use st.multiselect to let the user select songs
    selected_songs = st.multiselect("Select songs to refine recommendations:", df['title'].unique())
    if selected_songs:
        return generate_content_based_recommendations(selected_songs, user_id)
    else:
        return []

def get_final_recommendations(user_id, initial_recommendations, refined_recommendations):
    # Implement logic to combine and filter recommendations
    # Return the final list of recommendations
    # For example, you can remove duplicates and filter songs the user has already listened to
    listened_songs = df[df['user'] == user_id]['title'].unique()
    final_recommendations = (
        initial_recommendations.append(refined_recommendations)
        .drop_duplicates(subset='title')
        .loc[~initial_recommendations['title'].isin(listened_songs)]
    )
    return final_recommendations

if __name__ == "__main__":
    main()
