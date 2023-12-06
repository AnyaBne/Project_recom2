import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load your dataset
df = pd.read_csv('data.csv')  # Replace with the path to your dataset

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
    song_titles = []
    for index, row in recommended_songs.iterrows():
        song_id = row['song']
        score = algo.predict(user_id, song_id).est
        scores.append(score)
        song_titles.append(row['title'])

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
    login_container = st.empty()

    if user_id:
        login_container.empty()  # Clear the login section

        # Step 2: Generate Initial Recommendations
        initial_recommendations_container = st.empty()
        initial_recommendations = get_initial_recommendations(user_id)
        initial_recommendations_container.subheader("Initial Recommendations")
        initial_recommendations_container.write(initial_recommendations)
        
        explain_recommendations(user_id)



        # Placeholder for refined recommendations
        refined_container = st.empty()

        # Step 3: User Refines Recommendations
        selected_songs = st.multiselect("Select songs to refine recommendations:", df['title'].unique())

        if selected_songs:
            # Step 4: Generate Refined Recommendations
            refined_recommendations = generate_content_based_recommendations(selected_songs, user_id, n=10)

            # Step 5: Final Output
            final_recommendations_container = st.empty()
            final_recommendations = get_final_recommendations(user_id, initial_recommendations, refined_recommendations)
            final_recommendations_container.subheader("Final Recommendations")
            final_recommendations_container.write(final_recommendations)

            final_recommendations_df = pd.DataFrame(final_recommendations)

            # Appeler la fonction d'explication
            explanations = explain_content_based_selection(selected_songs, final_recommendations_df, tfidf_vectorizer)

            # Afficher les explications dans Streamlit
            st.subheader("Explanation for Recommendations:")
            for explanation in explanations:
               st.info(explanation)
        else:
            st.warning("Please select at least one song to refine recommendations.")


    # Step 5: Final Output
    #final_recommendations = get_final_recommendations(user_id, initial_recommendations, refined_recommendations)
    #st.subheader("Final Recommendations")
    #st.write(final_recommendations)

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
    recommended_songs = df[df['song'].isin(recommended_song_ids)][['title', 'artist_name']].drop_duplicates().head(n)
    
    return recommended_songs

def generate_content_based_recommendations(selected_songs, user_id, n=10):
    # Get the songs the user has already listened to
    user_songs = set(df[df['user'] == user_id]['title'].unique())

    # Exclude the selected songs from user_songs
    user_songs -= set(selected_songs)

    # Transform the selected songs' attributes into TF-IDF vectors
    selected_song_vectors = []
    for selected_song in selected_songs:
        selected_song_attributes = df[df['title'] == selected_song]['combined_attributes'].iloc[0]
        selected_song_vectors.append(tfidf_vectorizer.transform([selected_song_attributes]))

    # Aggregate TF-IDF vectors (you may choose a different strategy based on your requirements)
    selected_songs_vector = sum(selected_song_vectors)

    # Calculate cosine similarities between the selected songs and all songs
    cosine_similarities = cosine_similarity(selected_songs_vector, tfidf_matrix)

    # Get the indices of the top-N similar songs
    similar_song_indices = cosine_similarities.argsort()[0][::-1]

    # Initialize a set to track unique song titles
    unique_song_titles = set(user_songs).union(set(selected_songs))

    # Get the titles and artist names of recommended songs that the user hasn't heard before
    recommended_songs = []

    for idx in similar_song_indices:
        song_title = df['title'].iloc[idx]
        artist_name = df['artist_name'].iloc[idx]
        if song_title not in unique_song_titles:
            unique_song_titles.add(song_title)
            recommended_songs.append({'title': song_title, 'artist_name': artist_name})
        if len(recommended_songs) >= n:
            break

    return recommended_songs

def get_final_recommendations(user_id, initial_recommendations, refined_recommendations, initial_weight=0.3, refined_weight=0.7, n=10):
    # Implement logic to combine and filter recommendations
    # Return the final list of recommendations
    # For example, you can remove duplicates and filter songs the user has already listened to
    listened_songs = df[df['user'] == user_id]['title'].unique()

    # Convert refined_recommendations to a DataFrame
    refined_df = pd.DataFrame(refined_recommendations)

    # Assign weights to recommendations
    initial_recommendations['weight'] = initial_weight
    refined_df['weight'] = refined_weight

    # Concatenate and calculate a weighted score
    combined_recommendations = pd.concat([initial_recommendations[['title', 'artist_name', 'weight']], 
                                          refined_df[['title', 'artist_name', 'weight']]])

    # Group by title and artist, sum the weights
    combined_recommendations = combined_recommendations.groupby(['title', 'artist_name']).agg({'weight': 'sum'}).reset_index()

    # Sort by the weighted score in descending order
    combined_recommendations = combined_recommendations.sort_values(by='weight', ascending=False)

    # Drop duplicates based on 'title'
    combined_recommendations = combined_recommendations.drop_duplicates(subset='title')

    # Filter out songs the user has already listened to
    combined_recommendations = combined_recommendations.loc[~combined_recommendations['title'].isin(listened_songs)]

    # Select only the first n recommendations
    final_recommendations = combined_recommendations.head(n)[['title', 'artist_name']]

    return final_recommendations


if __name__ == "__main__":
    main()
