import os
import pandas as pd

FILENAME = "./data/spotify_top_songs_data.csv"

# Define a function to recommend songs based on the predicted emotion
def recommend_songs(df, emotion):
    # Setting thresholds for each emotion
    if emotion == 'Happy':
        # Songs with high valence and danceability
        recommended_songs = df[(df['valence'] > 0.7) & (df['danceability'] > 0.7)]
    elif emotion == 'Sad':
        # Songs with low valence, energy, and tempo
        recommended_songs = df[(df['valence'] < 0.3) & (df['energy'] < 0.3) & (df['tempo'] < 100)]
    elif emotion == 'Angry':
        # Songs with high energy and tempo
        recommended_songs = df[(df['energy'] > 0.7) & (df['tempo'] > 120)]
    elif emotion == 'Neutral':
        # Songs with average valence and energy
        recommended_songs = df[(df['valence'].between(0.4, 0.6)) & (df['energy'].between(0.4, 0.6))]
    elif emotion == 'Fear':
        # Songs with low loudness and energy
        recommended_songs = df[(df['loudness'] < -10) & (df['energy'] < 0.4)]
    elif emotion == 'Disgust':
        # Songs with low danceability and valence
        recommended_songs = df[(df['danceability'] < 0.3) & (df['valence'] < 0.3)]
    elif emotion == 'Surprise':
        # Songs with high liveness and speechiness
        recommended_songs = df[(df['liveness'] > 0.7) & (df['speechiness'] > 0.5)]

    # Sort by popularity and return top songs, considering max_value as a filter for confidence
    return recommended_songs.sort_values(by='popularity', ascending=False).head(10)


def get_song_information(predicted_emotion, prob, file_name):
    
    songs_df = pd.read_csv(file_name)
    
    recommended_songs = recommend_songs(songs_df, predicted_emotion)
    n_get_song = (10 - int(round(10 * prob)))
    
    artist = recommended_songs.iloc[n_get_song, 2]
    title = recommended_songs.iloc[n_get_song, 3]
    uri = recommended_songs.iloc[n_get_song, 4]
    
    return artist, title, uri
    

if __name__ == "__main__":
    # Using the given emotions and max_value
    predicted_emotion = "Neutral"  # Example emotion
    max_value = 0.36911362409591675  # Example value from the given code
    
    songs_df = pd.read_csv(FILENAME)

    # Get the recommended songs
    recommended_songs = recommend_songs(songs_df, predicted_emotion)
    n_get_song = int(round(10 * max_value)) - 1
    
    artist = recommended_songs.iloc[n_get_song, 2]
    title = recommended_songs.iloc[n_get_song, 3]
    uri = recommended_songs.iloc[n_get_song, 4]
    
    print("")