from flask import Flask, request, jsonify
import time

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Simulate model computation time
    time.sleep(0.1)
    return jsonify({'result': 'ok'})

if __name__ == '__main__':
    app.run(debug=True, port=5001)