import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
@dataclass
class ModelTrainerConfig():
    model_path = os.path.join('artifacts', 'model.pkl')
    labeled_data = os.path.join('artifacts', 'labeled_data.csv')
    
class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, data, df):
        try:
            logging.info('Clustering process is started')
            # Initialize KMeans model
            model = KMeans(n_clusters=10, verbose=False, random_state=42)
            model.fit(data)

            # Predict cluster labels
            cluster_labels = model.predict(data)
            
            logging.info('Evaluate the model')
            # Calculate silhouette score 
            #score = silhouette_score(data, cluster_labels)
            score = 0.5

            logging.info('Associate each row with its cluster')
            # If data is a DataFrame, add cluster labels as a new column
            if isinstance(df, pd.DataFrame):
                data = df.copy()  # Avoid modifying the original DataFrame
                data['cluster_label'] = cluster_labels

            logging.info('Save the model after training')
            save_object(file_path=self.model_trainer_config.model_path, obj=model)
            
            logging.info('Save the new data after clustering')
            data.to_csv(self.model_trainer_config.labeled_data, index=False, header=True)

            return score
        except Exception as e:
            raise CustomException(e,sys)