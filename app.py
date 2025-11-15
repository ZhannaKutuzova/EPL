from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

# Suppress warnings from scikit-learn when loading the model
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

app = Flask(__name__)

# Define the path to the models directory
models_dir = 'models'

# Load the preprocessor and model from the 'models' directory
try:
    preprocessor = joblib.load(os.path.join(models_dir, 'preprocessing_pipeline_final.pkl'))
    model = joblib.load(os.path.join(models_dir, 'model_totals_final.pkl'))
    # Get the index of 'Over' and 'Under' in model.classes_
    over_idx = list(model.classes_).index('Over')
    under_idx = list(model.classes_).index('Under')
except FileNotFoundError:
    print(f"Error: Model files not found in {models_dir}. Please ensure they are present.")
    exit(1)
except Exception as e:
    print(f"Error loading models: {e}")
    exit(1)

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Premier League Over/Under 2.5 Goals Predictor API is running!'
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Expected features
    expected_features = [
        'HomeTeam', 'AwayTeam', 'AvgH', 'AvgD', 'AvgA', 'Avg>2.5', 'Avg<2.5'
    ]

    # Validate input data
    for feature in expected_features:
        if feature not in data:
            return jsonify({'error': f'Missing data for key: {feature}'}), 400

    try:
        input_df = pd.DataFrame({
            'HomeTeam': [str(data['HomeTeam'])],
            'AwayTeam': [str(data['AwayTeam'])],
            'AvgH': [float(data['AvgH'])],
            'AvgD': [float(data['AvgD'])],
            'AvgA': [float(data['AvgA'])],
            'Avg>2.5': [float(data['Avg>2.5'])],
            'Avg<2.5': [float(data['Avg<2.5'])]
        })
    except ValueError as e:
        return jsonify({'error': f'Invalid data type for feature: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during data parsing: {e}'}), 500

    # Preprocess the input data
    try:
        processed_data = preprocessor.transform(input_df)
    except Exception as e:
        return jsonify({'error': f'Error during preprocessing: {e}'}), 500

    # Check for unknown categorical values after preprocessing (OrdinalEncoder returns -1)
    # Categorical features are typically at the end of the transformed array
    # In our case, HomeTeam and AwayTeam are the last two features (index 5 and 6)
    if (processed_data[0, 5] == -1.0 or processed_data[0, 6] == -1.0):
        return jsonify({'error': 'One or both teams not recognized. Please ensure team names are valid.'}), 400

    # Make prediction
    prediction_proba = model.predict_proba(processed_data)[0]
    predicted_class = model.predict(processed_data)[0]

    prob_over = prediction_proba[over_idx]
    prob_under = prediction_proba[under_idx]

    # Get odds from input data
    odds_over = float(data['Avg>2.5'])
    odds_under = float(data['Avg<2.5'])

    # Calculate Expected Value (EV) using the specified formula: EV = (probability * odds) - 1
    ev_over = (prob_over * odds_over) - 1
    ev_under = (prob_under * odds_under) - 1

    # Identify value bet
    value_bet_recommendation = "None"
    if ev_over > 0.05 and (ev_over >= ev_under or ev_under <= 0.05):
        value_bet_recommendation = f"Over 2.5 Goals @ odds {odds_over:.2f} (EV: {ev_over:.2%})"
    elif ev_under > 0.05 and (ev_under > ev_over or ev_over <= 0.05):
        value_bet_recommendation = f"Under 2.5 Goals @ odds {odds_under:.2f} (EV: {ev_under:.2%})"

    return jsonify({
        'prediction': predicted_class,
        'probabilities': {
            'Over': f'{prob_over:.2%}',
            'Under': f'{prob_under:.2%}'
        },
        'expected_value': {
            'Over': f'{ev_over:.2%}',
            'Under': f'{ev_under:.2%}'
        },
        'value_bet_recommendation': value_bet_recommendation
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
