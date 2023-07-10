from flask import Flask, jsonify
import pandas as pd
import retrieve_train_predict
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('trained_model.h5')

# Flask routes
@app.route('/prediction/monthly')
def get_monthly_prediction():
    # Retrieve trainPredict from the shared location using the retrieve_train_predict module
    trainPredict = retrieve_train_predict.retrieve_train_predict()

    if trainPredict is None:
        return jsonify({'error': 'Failed to retrieve trainPredict from shared location'})

    # Make predictions using the loaded model
    trainX = np.load('trainX.npy')
    testX = np.load('testX.npy')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Process the trainPredict variable and generate the desired response
    df = pd.DataFrame(trainPredict[:30])

    high_production_days = df.nlargest(5, 0).index.tolist()
    low_production_days = df.nsmallest(5, 0).index.tolist()

    response = {
        'high_production_days': high_production_days,
        'low_production_days': low_production_days
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=6000)
