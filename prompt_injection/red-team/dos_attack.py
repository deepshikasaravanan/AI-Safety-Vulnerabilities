import requests
import threading

def send_request():
    try:
        requests.post('http://127.0.0.1:5001/predict', json={'input': [0]*64})
    except Exception as e:
        print("Request failed:", e)

threads = []
for i in range(200):
    t = threading.Thread(target=send_request)
    threads.append(t)
    t.start()

for t in threads:
    t.join()

print("Attack finished.")