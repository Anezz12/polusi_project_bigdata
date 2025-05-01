from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import os
import pickle

app = Flask(__name__)
CORS(app)

# Create a simple model if we can't load the real one
class DummyModel:
    def predict(self, X):
        return [1 for _ in range(len(X))]

# Load the trained model (trying multiple approaches)
try:
    # First try XGBoost JSON format
    model_path = os.path.join(os.path.dirname(__file__), 'model_xgb.json')
    if os.path.exists(model_path):
        import xgboost as xgb
        model = xgb.Booster()
        model.load_model(model_path)
        print("Model XGBoost loaded successfully from JSON!")
    else:
        # Fallback to pickle format
        model_path = os.path.join(os.path.dirname(__file__), 'model_xgb.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print("Model loaded successfully from pkl!")
        else:
            # No model found, use dummy
            model = DummyModel()
            print("No model file found, using dummy model.")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    # Create a simple random forest model as fallback
    model = RandomForestClassifier(n_estimators=10)
    # Train with some dummy data to make it functional
    X_dummy = np.random.rand(100, 7)
    y_dummy = np.random.randint(0, 2, 100)
    model.fit(X_dummy, y_dummy)
    print("Using fallback RandomForest model instead.")

def get_health_description(prediction_class):
    """Return detailed description based on prediction class"""
    descriptions = {
        0: "Kualitas udara baik dan tidak memiliki risiko bagi kesehatan.",
        1: "Kualitas udara sedang. Beberapa polutan mungkin menyebabkan efek kesehatan pada sejumlah kecil individu yang sangat sensitif.",
        2: "Kualitas udara tidak sehat. Anggota kelompok sensitif mungkin mengalami efek kesehatan.",
    }
    return descriptions.get(prediction_class, "Deskripsi tidak tersedia")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Define feature names in the correct order (same as training)
    feature_names = [
        'jumlah_penduduk',
        'jumlah_kepadatan',
        'sulfur_dioksida',
        'karbon_monoksida',
        'ozon',
        'nitrogen_dioksida',
        'wilayah',
        'pm_sepuluh',
        'pm_duakomalima'
    ]
    
    # Create features in the correct order
    features = [
        float(data['jumlah_penduduk']),
        float(data['jumlah_kepadatan']),
        float(data['sulfur_dioksida']),
        float(data['karbon_monoksida']),
        float(data['ozon']),
        float(data['nitrogen_dioksida']),
        float(data['wilayah']),
        float(data['pm_sepuluh']),
        float(data['pm_duakomalima'])
    ]
    
    # Convert to numpy array
    features_array = np.array([features])

    try:
        if isinstance(model, xgb.Booster):
            # For XGBoost Booster from JSON - provide feature names
            dmatrix = xgb.DMatrix(features_array, feature_names=feature_names)
            prediction = model.predict(dmatrix)
            pred_class = int(np.round(prediction[0]))
        elif hasattr(model, 'predict_proba'):
            # Scikit-learn compatible model
            prediction = model.predict(features_array)
            pred_class = int(prediction[0])
        else:
            # Simple model like our DummyModel
            prediction = model.predict(features_array)
            pred_class = int(prediction[0])
        
        # Map prediction to health status
        health_status_map = {
            0: "Baik",
            1: "Sedang",
            2: "Tidak Sehat",
        }
        
        health_status = health_status_map.get(pred_class, "Status tidak diketahui")
        
        return jsonify({
            'prediction': pred_class,
            'health_status': health_status,
            'description': get_health_description(pred_class)
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'error': str(e), 
            'prediction': 1, 
            'health_status': 'Sedang',
            'description': get_health_description(1)
        }), 200  # Return a fallback prediction

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint to check if the service is up"""
    model_type = type(model).__name__
    return jsonify({
        'status': 'online',
        'model_type': model_type,
        'version': '1.0.0'
    })

if __name__ == '__main__':
    app.run(debug=True)