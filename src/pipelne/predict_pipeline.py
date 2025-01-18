import os
import sys
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline():
    def __init__(self):
        try:
            logging.info("Load the dataset...")
            self.data = pd.read_csv('artifacts\labeled_data.csv')
            logging.info("The dataset loaded successfully.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, song):
        try:
            pass
        except Exception as e:
            raise CustomException(e, sys)