from flask import Flask, request, jsonify, send_from_directory
import pickle
import pandas as pd
import numpy as np
import os
app = Flask(__name__)

MODEL_PATH   = 'LinearRegressionModel.pkl'
COLUMNS_PATH = 'feature_columns.pkl'

if not os.path.exists(MODEL_PATH) or not os.path.exists(COLUMNS_PATH):
    raise FileNotFoundError(
        "❌  Model files not found!\n"
        "    Run  car_price_predictor.py  first to train and save the model."
    )

model         = pickle.load(open(MODEL_PATH,   'rb'))
feature_cols  = pickle.load(open(COLUMNS_PATH, 'rb'))

print("✅  Model loaded successfully.")
print(f"    Features: {len(feature_cols)} columns")

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        name       = str(data['name'])       
        company    = str(data['company'])     
        year       = int(data['year'])        
        kms_driven = int(data['kms_driven'])  
        fuel_type  = str(data['fuel_type'])   

        sample = pd.DataFrame(
            [[name, company, year, kms_driven, fuel_type]],
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']
        )
        sample = pd.get_dummies(sample, drop_first=True)
        sample = sample.reindex(columns=feature_cols, fill_value=0)
        price = model.predict(sample)[0]
        price = max(float(price), 10000)  

        return jsonify({
            'success': True,
            'price': round(price, 2),
            'price_formatted': f"₹{price:,.0f}",
            'input': {
                'name': name,
                'company': company,
                'year': year,
                'kms_driven': kms_driven,
                'fuel_type': fuel_type
            }
        })

    except KeyError as e:
        return jsonify({'success': False, 'error': f'Missing field: {e}'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    return jsonify({
        'total_features': len(feature_cols),
        'feature_names': list(feature_cols),
        'model_type': str(type(model).__name__)
    })

if __name__ == '__main__':
    print("\n🚀  Starting Car Price Predictor server...")
    print("    Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
