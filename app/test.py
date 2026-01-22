import requests

url = "http://localhost:8000/analyze"
files = {"file": open("sample_audio.wav", "rb")}

response = requests.post(url, files=files)
print(response.json())
