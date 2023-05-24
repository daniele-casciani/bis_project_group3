import requests
import json
### The code provided demonstrates how to send a POST request to the /processing/start route 
### of a Flask server running on http://localhost:5000. It uses the requests library to make the 
### HTTP request and includes sample JSON payloads.

# Assuming your Flask server is running on http://localhost:5000

# Sample JSON payloads
json_payloads = [
    {

            "event_id": "gr1-Turkey-2023_02_06T00_00_00-2023_02_06T12_00_00-earthquake",

            "country": "Turkey",

            "date": "2023-02-06 01:17:34",

            "type": "earthquake",

            "timeframe": [

                "2023-02-06 00:00:00",

                "2023-02-06 12:00:00"

            ],

            "locations": [

                [

                    37.0143,

                    37.2256

                ],

                [

                    37.0001,

                    37.2252

                ],

                [

                    36.8929,

                    37.1893

                ],

                [

                    37.9141,

                    37.7712

                ],

                [

                    38.0613,

                    37.9227

                ],

                [

                    36.9658,

                    37.1816

                ],

                [

                    38.5342,

                    38.1845

                ],

                [

                    37.1962,

                    38.0106

                ],

                [

                    38.0984,

                    38.0315

                ],

                [

                    37.2033,

                    37.9962

                ],

                [

                    37.8023,

                    38.0249

                ],

                [

                    38.1847,

                    38.248

                ]

            ],

            "images": [

                {

                    "tweet_url": "https://twitter.com/GeorgeMelikyan/status/1622416823613276162",

                    "image_url": "https://pbs.twimg.com/ext_tw_video_thumb/1622416565995028481/pu/img/FkoHSJl13HCLTd1g.jpg",

                    "date": "2023-02-06 02:08:08",

                    "timeframe": [

                        "2023-02-06 02:00:00",

                        "2023-02-06 02:15:00"

                    ]

                },

                {

                    "tweet_url": "https://twitter.com/deZabedrosky/status/1622430065786077184",

                    "image_url": "https://pbs.twimg.com/ext_tw_video_thumb/1622406738505785346/pu/img/rqMhxZ-NeAfPzB-x.jpg",

                    "date": "2023-02-06 03:00:45",

                    "timeframe": [

                        "2023-02-06 03:00:00",

                        "2023-02-06 03:15:00"

                    ]

                },

                {

                    "tweet_url": "https://twitter.com/Chrisweather_/status/1622449825361891329",

                    "image_url": "https://pbs.twimg.com/media/FoQZA9NWIAE2g0e?format=jpg&name=small",

                    "date": "2023-02-06 04:19:16",

                    "timeframe": [

                        "2023-02-06 04:15:00",

                        "2023-02-06 04:30:00"

                    ]

                },

                {

                    "tweet_url": "https://twitter.com/oguzcemcelik/status/1622494708378255362",

                    "image_url": "https://pbs.twimg.com/ext_tw_video_thumb/1622486488272232450/pu/img/_I2gWEw7ZFFX1ulL.jpg",

                    "date": "2023-02-06 07:17:37",

                    "timeframe": [

                        "2023-02-06 07:15:00",

                        "2023-02-06 07:30:00"

                    ]

                },

                {

                    "tweet_url": "https://twitter.com/faizan_wadood3/status/1622534860563968001",

                    "image_url": "https://pbs.twimg.com/media/FoRm2MJagAA8zbZ?format=jpg&name=small",

                    "date": "2023-02-06 09:57:10",

                    "timeframe": [

                        "2023-02-06 09:45:00",

                        "2023-02-06 10:00:00"

                    ]

                },

                {

                    "tweet_url": "https://twitter.com/HumanDilemma_/status/1622560743265648642",

                    "image_url": "https://pbs.twimg.com/ext_tw_video_thumb/1622559844782727168/pu/img/IcvoAEOS_PVspCQX.jpg",

                    "date": "2023-02-06 11:40:01",

                    "timeframe": [

                        "2023-02-06 11:30:00",

                        "2023-02-06 11:45:00"

                    ]

                }

            ]

        }
]

# Create a list of file objects with the JSON data
file_objs = [('files', json.dumps(payload)) for payload in json_payloads]

# Send the POST request to start crawling
response = requests.post('http://3.69.174.135:3333/processing/start', files=file_objs)

# Send post to the localhost 
#response = requests.post('http://localhost:5000/processing/start', files=file_objs)

# Check the response
if response.status_code == 200:
    print(response)
else:
    print(f"Error: {response.status_code} - {response.text}")
