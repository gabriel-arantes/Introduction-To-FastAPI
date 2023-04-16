# -*- coding utf-8 -*-

# 1. Library Imports
import uvicorn
from fastapi import FastAPI
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

# 2. Create the app object
app = FastAPI()
with open('random_forest_model.pkl', 'rb') as pickle_in:
    classifier = pickle.load(pickle_in)


# 3. Index route, opens automatically on http://127.0.0.1:8000
@app.get('/')
def index():
    return {'message': 'Hello, friend'}

# 4. Route with  a single parameter, returns the parameter within a message
#    Located at: http://127.0.0.1:8000/AnyNameHere
@app.get('/{name}')
def get_name(name: str):
    return {'message:', f'Hello, {name}'}

# 5. Expose the prediction functionalitym make a prediction
#    JSON data and return the predicted Bank Note with the confidence
@app.post('/predict')
def predict_banknote(data:BankNote):
    
    data = data.dict()
    
    variance = data['variance']
    skewness = data['skewness']
    curtosis = data['curtosis']
    entropy = data['entropy']

    print(classifier.predict([[variance, skewness, curtosis, entropy]]))
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])

    if(prediction[0] > 0.5):
        prediction = "Fake Note"

    else:
        prediction = "Its a Bank Note"
    
    return {
        'prediction': prediction
    }

# 6. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port = 8000)

#uvicorn app:app --reload
