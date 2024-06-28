from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# curl -X POST -H "Content-Type: application/json" -d  @data.json http://127.0.0.1:5000/predict

app = Flask(__name__)

features_names = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z']

# Load the pre-trained model
with open('../models/linearxbg_06_27_2024_22_34_07/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-trained model
with open('../models/linearxbg_06_27_2024_22_34_07/preprocessor.pkl', 'rb') as model_file:
    preprocessor = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (assuming JSON format)
        data = request.get_json()

        data_array = np.array(list(data.values())).reshape(1, -1)
        df = pd.DataFrame(data_array, columns=list(features_names))
        
        print(data.keys())
        print(df)

        # transform data using the preprocessor
        processed_data = preprocessor.transform(df)
        print("processed data", processed_data)

        # Make predictions
        prediction = model.predict(processed_data)[0]
        print("prediction", prediction)

        return jsonify({'predicted_value': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/debug/<input_string>')
def debug_route(input_string):
    return f"Received string: {input_string}"

app.run()

if __name__ == '__main__':
    app.run(debug=True)