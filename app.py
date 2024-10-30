from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler once at startup
model = joblib.load('model/xgboost_model.pkl')
scaler = joblib.load('utils/scaler.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json  # Get JSON data from request
    input_data['N'] = float(input_data['N'])
    input_data['P'] = float(input_data['P'])
    input_data['K'] = float(input_data['K'])
    input_data['temperature'] = float(input_data['temperature'])
    input_data['humidity'] = float(input_data['humidity'])
    input_data['ph'] = float(input_data['ph'])
    input_data['rainfall'] = float(input_data['rainfall'])

    # Create a DataFrame from input data
    df = pd.DataFrame([input_data])

    # Scale the data
    df_scaled = scaler.transform(df)

    # Make prediction
    prediction = model.predict(df_scaled)[0]
    
    

    return jsonify({"predicted_crop": prediction})

if __name__ == '__main__':
    app.run(debug=True,port=5001,host='0.0.0.0')
