from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)

# Load the keras model
model = tf.keras.models.load_model('HPF.keras')

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.json
        
        # Extract PPG, ABP and ECG signals
        ppg_signal = np.array(data["PPG"]).flatten()
        ecg_signal = np.array(data["ECG"]).flatten()
        
        # We don't need ABP for prediction, but let's validate it exists
        if "ABP" not in data:
            return jsonify({'error': 'Missing ABP data'}), 400
        
        if len(ppg_signal) == 0 or len(ecg_signal) == 0:
            return jsonify({'error': 'Missing required input data (PPG or ECG)'}), 400
            
        # Z-Score Normalization
        def z_score_normalize(signal):
            return (signal - np.mean(signal)) / np.std(signal)

        ppg_signal = z_score_normalize(ppg_signal)
        ecg_signal = z_score_normalize(ecg_signal)

        # Segment signals into 10-second windows
        fs = 125  # Sampling frequency
        window_size = 10 * fs  # 1250 samples per window

        def segment_signal(signal, window_size):
            # If signal length is not exactly divisible by window_size,
            # we'll use as many complete segments as possible
            num_segments = len(signal) // window_size
            segments = []
            for i in range(num_segments):
                segments.append(signal[i * window_size:(i + 1) * window_size])
            return np.array(segments)

        ppg_segments = segment_signal(ppg_signal, window_size)
        ecg_segments = segment_signal(ecg_signal, window_size)

        # Combine PPG & ECG into model input format - note we're only using PPG and ECG
        dataset_X = np.stack((ppg_segments, ecg_segments), axis=-1)
        
        # Make Predictions
        predictions_HPF = model.predict(dataset_X)

        # Denormalization Functions
        def denormalize_sbp(value):
            return np.clip(value * 160 + 40, 90, 180)  # Allow SBP up to 180

        def denormalize_dbp(value):
            return np.clip(value * 160 + 40, 60, 120)  # Allow DBP down to 60

        # Apply denormalization
        predicted_sbp = denormalize_sbp(predictions_HPF[:, 0])
        predicted_dbp = denormalize_dbp(predictions_HPF[:, 1])
        
        # Create response with proper serialization
        response = {
            'sbp': predicted_sbp.tolist(),
            'dbp': predicted_dbp.tolist()
        }
        
        return app.response_class(
            response=json.dumps(response, cls=NumpyEncoder),
            status=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)