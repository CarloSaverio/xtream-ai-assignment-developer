from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

# example of post > curl -X POST -H "Content-Type: application/json" -d  @data.json http://127.0.0.1:5000/predict
# example of get  > curl -X GET "http://127.0.0.1:5000/get_similar_diamonds?cut=Ideal&color=H&clarity=SI2&carat=1.10"

app = Flask(__name__)

# Load the pre-trained model
with open('../models/linearxbg_06_27_2024_22_34_07/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the pre-trained model
with open('../models/linearxbg_06_27_2024_22_34_07/preprocessor.pkl', 'rb') as model_file:
    preprocessor = pickle.load(model_file)

training_data = pd.read_csv('../data/diamonds.csv')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (assuming JSON format)
        data = request.get_json()

        data_array = np.array(list(data.values())).reshape(1, -1)
        df = pd.DataFrame(data_array, columns=list(data.keys()))

        # transform data using the preprocessor
        processed_data = preprocessor.transform(df)
        print("processed data", processed_data)

        # Make predictions
        prediction = model.predict(processed_data)[0]
        print("prediction", prediction)

        return jsonify({'predicted_value': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/get_similar_diamonds', methods=['GET'])
def get_similar_diamonds():
    try:
        print(request.args)
        cut = request.args.get('cut')           # Get cut value from query parameter
        color = request.args.get('color')       # Get color value from query parameter
        clarity = request.args.get('clarity')   # Get clarity value from query parameter
        carat = request.args.get('carat')       # get the reference of carat 

        n = 5   # if it should vary the user must provide it in the future

        # Filter Dataset based on cut, color, and clarity
        similar_diamonds = training_data[
            (training_data['cut'] == cut) &
            (training_data['color'] == color) &
            (training_data['clarity'] == clarity)
        ]

        # Calculate error w.r.t carat setpoint and sort it
        similar_diamonds['diff'] = (similar_diamonds['carat'] - float(carat)).abs()
        closest_row = similar_diamonds.sort_values(by='diff', ascending=True)
        
        # Select top n samples
        selected_samples = closest_row.head(n)

        # Convert selected samples to JSON format
        result = selected_samples.to_dict(orient='records')

        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})		
		
# JUST TO DEBUG		
@app.route('/debug/<input_string>')
def debug_route(input_string):
    return f"Received string: {input_string}"

app.run()

if __name__ == '__main__':
    app.run(debug=True)