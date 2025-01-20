import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging


class PredictPipeline():
    def __init__(self):
        try:
            logging.info("Load the dataset...")
            self.data = pd.read_csv("artifacts\labeled_data.csv")
            logging.info("The dataset loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def find(self, song_name):
        try:
            logging.info('Searching for the song in the dataset...')
            # Perform case-insensitive search for the song
            row = self.data[self.data['name'].str.lower() == song_name.lower()]
            if row.empty:
                logging.info("Song not found in the dataset. Please try another song.")
                return None  # Return None if song is not found
            logging.info('Song found in the dataset.')
            return row.iloc[0]  # Return the first matching row as a Series
        except Exception as e:
            raise CustomException(e, sys)
        
    def recommend(self, song_name, num_recommendations = 5):
        try:
            # Find the song in the dataset
            logging.info("Finding the input song...")
            row = self.find(song_name)

            if row is None:
                raise ValueError(f"Song '{song_name}' not found in the dataset. Unable to provide recommendations.")

            logging.info("Retrieving recommendations...")
            # Filter songs in the same cluster
            recommendations = self.data[self.data['cluster_label'] == row['cluster_label']]

            logging.info("Excluding the input song from recommendations...")
            # Exclude the input song from recommendations
            recommendations = recommendations[recommendations['name'].str.lower() != song_name.lower()]

            logging.info("Limiting the number of recommendations...")
            # Limit the number of recommendations
            recommendations = recommendations.sample(min(num_recommendations, len(recommendations)))

            logging.info("Recommendations generated successfully.")
            return recommendations[['name']]  
        except Exception as e:
            raise CustomException(e, sys)
        
        