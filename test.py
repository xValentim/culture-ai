import requests
import json

url = "http://localhost:8000/chat"

messages = [
    {
      "type": "ai",
      "content": "Olá, em que posso ajudá-lo?"
    },
    {
        "type": "human",
        "content": "Olá, quero ver 3D da obra venus",
    }
]

response = requests.post(url, json=messages, stream=True)

for line in response.iter_lines():
    if line:
        chunk = line.decode('utf-8').replace('data: ', '')
        chunk_json = json.loads(chunk)
        print(chunk_json['content'], end='', flush=True)