import requests

# Send invalid input (wrong length)
response = requests.post('http://127.0.0.1:5002/predict', json={'input': [1, 2, 3]})
print("Response to short input:", response.json())

# Send input that triggers a different error
response = requests.post('http://127.0.0.1:5002/predict', json={'input': "not a list"})
print("Response to wrong type:", response.json())