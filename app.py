import sys
from flask import Flask, render_template, request
import pandas as pd

from src.pipelne.predict_pipeline import PredictPipeline
from src.utils import convert_to_dict
from src.exception import CustomException
from src.logger import logging


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('website.html')

@app.route('/getRecommendations', methods=['POST'])
def getRecommendations():
    try:
        song_name = request.form.get('song')
        predict_pipeline = PredictPipeline()
        recommendations = predict_pipeline.recommend(song_name)
        recommendations = recommendations.to_dict(orient='records')
        recommendations = convert_to_dict(recommendations)
        return render_template('website.html', recommendations = recommendations)
    except Exception as e:
        raise CustomException(e,sys)
        


if __name__ == '__main__':
    app.run(debug=True)