import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

    # Step 3: User Refines Recommendations
    refined_recommendations = get_refined_recommendations(user_id)
    st.subheader("Refined Recommendations")
    st.write(refined_recommendations)

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
