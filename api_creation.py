from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import json

from textblob import TextBlob
# Create Flask app
app = Flask(__name__)

import pickle

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    ''' input : Json of "country": "Armenia",
        "points": 87,
        "price": 36.781224,
        "review_description": "not good"

        output : {
    "prediction": "['Red Blend']"
        }
    '''

    input_data = request.json[0]
    country_map = {'Armenia': 0, 'Slovakia': 1, 'India': 2, 'Switzerland': 3, 'Luxembourg': 4, 'Czech Republic': 5, 'Serbia': 6, 'Macedonia': 7, 'Cyprus': 8, 'Ukraine': 9, 'Peru': 10, 'Croatia': 11, 'Georgia': 12, 'Morocco': 13, 'Lebanon': 14, 'Brazil': 15, 'Turkey': 16, 'Uruguay': 17, 'Moldova': 18, 'Hungary': 19, 'Mexico': 20, 'Slovenia': 21, 'England': 22, 'Romania': 23, 'Bulgaria': 24, 'Greece': 25, 'Canada': 26, 'Israel': 27, 'South Africa': 28, 'Australia': 29, 'New Zealand': 30, 'Germany': 31, 'Austria': 32, 'Argentina': 33, 'Spain': 34, 'Chile': 35, 'Portugal': 36, 'Italy': 37, 'France': 38, 'US': 39}
    if input_data['country'] not in country_map: 
        return jsonify({'prediction': "Country doesnt exist"})
    input_data['country'] = country_map[input_data['country']]

    input_data['review_sentiment'] = TextBlob(input_data['review_description']).sentiment.polarity
    print(input_data)
    input_data = pd.DataFrame(input_data,index=[0])
    input_data.drop('review_description',inplace=True, axis=1)

    
    
    prediction = model.predict(input_data)
    
    return jsonify({'prediction': str(prediction)})

@app.route('/', methods=['GET'])
def index():
    return 'Machine Learning Inference'


if __name__ == '__main__':
    app.run()
