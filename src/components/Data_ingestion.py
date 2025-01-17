import os 
import sys
import pandas as pd
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging

@dataclass
class DataIngestionConfig:
    data_path = os.path.join('artifacts', 'data.csv')
    

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig
        
    def initiate_data_ingestion(self):
        logging.info('Entered the data ingestion method')
        try:
            df = pd.read_csv('notebook\Data\data.csv')
            logging.info('Read the dataset as DataFrame')
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Data ingestion is completed')
            return self.ingestion_config.data_path
        except Exception as e:
            raise CustomException(e,sys)   
     

if __name__ == "__main__":
    obj = DataIngestion()
    data = obj.initiate_data_ingestion()
    
    