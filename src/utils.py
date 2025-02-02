import os
import sys
import pandas as pd
import numpy as np
import dill

from src.exception import CustomException
from src.logger import logging

# function to remove unnecessary columns from DataFrame:
def drop_columns(df, columns):
    try:
        X = df.drop(columns = columns,  errors='ignore')
        return X
    except Exception as e:
        raise CustomException(e,sys)
    

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
    except Exception as e:
        raise CustomException(e, sys)
    
def convert_to_dict(list1):
    try: 
        list2 = [{'image_path': 'static\images\card.jpg'}, {'image_path': 'static\images\card02.jpg'}, {'image_path': 'static\images\card3.jpg'},{'image_path': 'static\images\card4.jpg'},{'image_path': 'static\images\card5.jpg'}]
        dict = [{**d1, **d2} for d1, d2 in zip(list1, list2)]
        return dict
    except Exception as e:
        raise CustomException(e, sys)