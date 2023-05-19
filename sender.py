import requests
import json

# Assuming your Flask server is running on http://localhost:5000

# Sample JSON payloads
json_payloads = [
{"id": "gr1-Syria-2023_02_06T00_00_00-2023_02_06T12_00_00-earthquake", "country": "Syria", "date": "2023-02-06 01:36:27", "type": "earthquake", "timeframe": ["2023-02-06 00:00:00", "2023-02-06 12:00:00"], "locations": [[36.6832, 36.9921], [36.6208, 37.0302], [36.7588, 38.0752]], "images": [{"URLTweet": "https://twitter.com/HumanDilemma_/status/1622560743265648642", "URLImage": "https://pbs.twimg.com/ext_tw_video_thumb/1622559844782727168/pu/img/IcvoAEOS_PVspCQX.jpg", "date": "2023-02-06 11:40:01"}, {"URLTweet": "https://twitter.com/AzadMirzaa/status/1622565757472088072", "URLImage": "https://pbs.twimg.com/media/FoSC8_qWAAExPhW?format=jpg&name=900x900", "date": "2023-02-06 11:59:57"}]}
]

# Create a list of file objects with the JSON data
file_objs = [('files', json.dumps(payload)) for payload in json_payloads]

# Send the POST request to start crawling
response = requests.post('http://3.69.174.135:3333/processing/start', files=file_objs)

# Check the response
if response.status_code == 200:
    print('ok')
else:
    print(f"Error: {response.status_code} - {response.text}")
