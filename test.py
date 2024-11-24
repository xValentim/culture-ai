import requests

url = "http://localhost:8000/chat"

messages = [
    {
      "type": "ai",
      "content": "Olá, em que posso ajudá-lo?"
    },
    {
        "type": "human",
        "content": "Olá, quero saber sobre parnasianismo",
    }
]

response = requests.post(url, json=messages, stream=True)

for line in response.iter_lines():
    if line:
        chunk = line.decode('utf-8').replace('data: ', '')
        print(chunk, end='', flush=True)