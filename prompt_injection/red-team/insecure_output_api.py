from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['input']
        arr = np.array(data)
        if arr.shape[0] != 64:
            raise ValueError("Input must have 64 features")
        pred = int(np.sum(arr) % 10)
        return jsonify({'prediction': pred})
    except Exception as e:
        # Insecure: returns internal error details to the user!
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5002)