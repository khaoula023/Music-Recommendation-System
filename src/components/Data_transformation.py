import os
import sys
from dataclasses import dataclass
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import drop_columns, save_object

@dataclass
class DataTransformationConfig:
    preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig
        
    def get_transformer(self):
        try:
            numerical_features = ['valence', 'year', 'acousticness', 'danceability', 'duration_ms', 'energy', 'explicit', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'popularity', 'speechiness', 'tempo']
            categorical_features = ['artists', 'name', 'id','release_date']
            
            logging.info('Building the transformer pipelines.')
            cat_pipeline = Pipeline([("drop_columns", FunctionTransformer(drop_columns, kw_args={'columns': categorical_features}))])
            
            logging.info("numerical columns encoding is started")
            num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
            
            preprocessor = ColumnTransformer([('categorical',cat_pipeline, categorical_features),
                                              ('numerical', num_pipeline, numerical_features)])
            
            logging.info('Data transformer pipeline is ready.')
            return preprocessor
            
        except Exception as e:
            raise CustomException(e,sys)    
        
    def transform(self, data_path):
        try:
            logging.info('Read the data as DataFrame')
            df = pd.read_csv(data_path)
            
            logging.info("Obtaining preprocessing object")
            preprocessor = self.get_transformer()
            
            logging.info("Applying preprocessing object on the dataframe")
            transformed_data = preprocessor.fit_transform(df)
            
            logging.info(f"Saved preprocessing object.")
            save_object(file_path= self.data_transformation_config.preprocessor_path,
                        obj= preprocessor)
            
            return transformed_data
      
        except Exception as e:
            raise CustomException(e,sys)