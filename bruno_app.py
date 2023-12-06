
import streamlit as st
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np


# Set Streamlit page configuration to wide mode
st.set_page_config(layout="wide")
# Add custom CSS for dark mode
st.markdown(
    """
    <style>
        body {
            color: white;
            background-color: #1E1E1E;
        }
    </style>
    """,
    unsafe_allow_html=True
)
# Load your dataset
dataset_url = 'https://raw.githubusercontent.com/AidaBJI/ML/main/data.csv'
df = pd.read_csv(dataset_url)  # Replace with the path to your dataset

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

# Explain function of the refine recommendation
def explain_content_based_selection(selected_songs, recommended_songs_df, df):
    explanations = []

    for song in selected_songs:
        selected_song_attributes = df[df['title'] == song]['combined_attributes2'].iloc[0]
        explanation = f"Selected song '{song}' has these key attributes: {selected_song_attributes}."
        explanations.append(explanation)

    explanations.append("\nBased on these attributes, the following songs are recommended:")

    for idx, rec_song in recommended_songs_df.iterrows():
        rec_song_title = rec_song['title']
        rec_song_attributes = df[df['title'] == rec_song_title]['combined_attributes2'].iloc[0]
        explanation = f"Recommended song '{rec_song_title}' has similar attributes: {rec_song_attributes}."
        explanations.append(explanation)

    return explanations



# Streamlit app
def main():
    st.title("Song Recommendation System")

    # Get the current state from the session
    state = get_state()

    if state["page"] == "login":
        show_login(state)
    elif state["page"] == "recommendations":
        # Obtain user_id from the session state
        #user_id = st.text_input("Enter your user ID:")
        show_recommendations(state)

def get_state():
    # This function creates a session state dictionary if it doesn't exist
    if "state" not in st.session_state:
        st.session_state.state = {"page": "login"}  # Default to the login page
    return st.session_state.state

def show_login(state):
    with st.form("login_form"):
        user_id = st.text_input("Enter your user ID:")
        login_button = st.form_submit_button("Login")

    if login_button:
        if user_id not in df['user'].unique():
            st.warning("Please enter a valid user ID")
        else:
            st.session_state.state = {"page": "recommendations", "user_id": user_id}
            st.experimental_rerun()

    # Load a song image for the background
    song_image_url = "https://github.com/AidaBJI/ML/blob/main/abstract-particles-wave-background_23-2148368029.jpg?raw=true"

    # Set the background image HTML
    background_image_html = f"""
        <style>
            body {{
                background-image: url('{song_image_url}');
                background-size: cover;
                background-position: center;
                opacity: 0.7;
                top: 0;
                left: 0;
                width: 150vw;
                height: 100vh;
            }}
        </style>
    """
    st.markdown(background_image_html, unsafe_allow_html=True)

    # Optionally display a message or other content in the login section
    pass
  # Display initial recommendations and scores


def show_recommendations(state):
    user_id = state.get("user_id")

    # Step 2: Generate Initial Recommendations
    initial_recommendations, scores = get_initial_recommendations(user_id, n=10)
    st.subheader("Try listening to something new!")
    state["initial_scores"] = scores
    # Container for horizontally scrollable images and text
    columns = st.columns(len(initial_recommendations))

    # Display initial recommendations as horizontally scrollable boxes
    for index, row in enumerate(initial_recommendations.itertuples()):
        with columns[index].container():
            display_horizontal_song_box(row.title, row.artist_name)

    # Placeholder for refined recommendations
    refined_container = st.empty()

    listened_songs = df[df['user'] == user_id]['title'].unique()

    # Step 3: User Refines Recommendations
    selected_songs = st.multiselect(
        "Pick a song you love, and we'll find similar tunes you might enjoy!",
        listened_songs  # Provide only the songs the user has listened to as options
    )

    if selected_songs:
        # Step 4: Generate Refined Recommendations
        refined_recommendations = generate_content_based_recommendations(selected_songs, user_id, n=10)

        # Display final recommendations outside the "About" section
        st.subheader("Final Recommendations")
        columns = st.columns(len(refined_recommendations))

        # Display refined recommendations as horizontally scrollable boxes
        for index, row in enumerate(refined_recommendations):
            with columns[index].container():
                display_horizontal_song_box(row['title'], row['artist_name'])
    else:
        st.warning("Select at least one song")

    # About button
    with st.expander("About"):
        st.write(
            "Welcome to the Song Recommendation System!",
            "To get an idea how this system works Imagine our recommendation system as your musical companion, suggesting songs",
            "it thinks you might like. It does this by looking at what other users who share similar",
            "musical tastes have enjoyed. So, for your initial recommendations, we start by ",
            "predicting how much you might enjoy certain songs based on what users with similar ",
            "tastes liked. The system assigns a score to each song, and the higher the score, the",
            "more confident it is that you'll enjoy the song. Here are the top songs it recommended ",
            "for you, sorted from the highest predicted enjoyment to the lowest:"
        )

        # Initialisation of titles and scores
        titles = []
        scores = []

        # Display initial recommendations and scores
        for (index, row), score in zip(enumerate(initial_recommendations.itertuples()), state["initial_scores"]):
            display_initial_recommendation_with_score(row.title, row.artist_name, user_id, score)
            titles.append(row.title)  # add titles to the list
            scores.append(score)
            

       
        plt.figure(figsize=(10, 6))
        plt.bar(titles, scores, color='skyblue') 
        plt.xlabel('Songs')
        plt.ylabel('Scores')
        plt.title('Histogram')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()  
        st.pyplot(plt)

        if selected_songs:
            explanations = explain_content_based_selection(selected_songs, refined_recommendations, tfidf_vectorizer)

            # display recommendation
            st.subheader("Explanation for refine recommendations:")
            for explanation in explanations:
               st.info(explanation)
            
    
        
    if st.button("Logout"):
        # Reset the state to go back to the login page
        st.session_state.state = {"page": "login"}
        st.experimental_rerun()
def display_initial_recommendation_with_score(title, artist_name, user_id, score):
    st.write(f"**Title:** {title}\n**Artist:** {artist_name}\n**Score:** {score:.2f}")

def display_horizontal_song_box(title, artist_name):
    # Add your music icon URL or local path
    music_icon_url = "https://www.iconfinder.com/icons/5584525/download/png/512"

    # Retrieve the image
    response = requests.get(music_icon_url)
    image = Image.open(BytesIO(response.content))

    # Display image and text in the container
    st.image(image, use_column_width=True)
    st.write(f"**Title:** {title}\n**Artist:** {artist_name}")

# Helper functions
def get_initial_recommendations(user_id, n=10):
    # Collaborative filtering logic
    listened_songs = df[df['user'] == user_id]['song'].unique()
    all_songs = df[~df['song'].isin(listened_songs)]['song'].unique()

    if len(all_songs) == 0:
        st.warning("No new songs found for recommendations. Try listening to more songs!")
        return pd.DataFrame(columns=['title', 'artist_name']), []

    predictions = []
    for song in all_songs:
        prediction = algo.predict(user_id, song)
        predictions.append((song, prediction.est))

    predictions.sort(key=lambda x: x[1], reverse=True)

    recommended_song_ids = [song for song, _ in predictions[:n]]
    recommended_songs = df[df['song'].isin(recommended_song_ids) & ~df['song'].isin(listened_songs)][['title', 'artist_name']].drop_duplicates().head(n)

    scores = [score for _, score in predictions[:n]]

    return recommended_songs, scores


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
