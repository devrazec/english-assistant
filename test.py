import requests

url = "http://localhost:8000/analyze"
files = {"file": open("sample_audio.mp3", "rb")}

response = requests.post(url, files=files)
print(response.json())
