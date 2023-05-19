import requests
import json

# Assuming your Flask server is running on http://localhost:5000

# Sample JSON payloads
json_payloads = [
    {'key1': 'value1'},
    {'key2': 'value2'}
]

# Create a list of file objects with the JSON data
file_objs = [('files', json.dumps(payload)) for payload in json_payloads]

# Send the POST request to start crawling
response = requests.post('http://localhost:5000/crowd4sdg/start', files=file_objs)

# Check the response
if response.status_code == 200:
    data = response.json()
    crawler_id = data['crawler_id']
    print(f"Crawler ID: {crawler_id}")
else:
    print(f"Error: {response.status_code} - {response.text}")
