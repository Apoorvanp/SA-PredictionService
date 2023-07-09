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

    # Process the trainPredict variable and generate the desired response
    column_names = ['MW Total']
    df = pd.DataFrame(trainPredict, columns=column_names)
    high_production_days = df.nlargest(5, 'MW Total')['MW Total'].index.tolist()
    low_production_days = df.nsmallest(5, 'MW Total')['MW Total'].index.tolist()

    # Make predictions using the loaded model
    trainX = np.load('trainX.npy')
    testX = np.load('testX.npy')
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    response = {
        'high_production_days': high_production_days,
        'low_production_days': low_production_days,
        'train_predictions': trainPredict.tolist(),
        'test_predictions': testPredict.tolist()
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(port=7000)
