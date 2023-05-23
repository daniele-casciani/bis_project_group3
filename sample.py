import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from process_image import process_image

app = Flask(__name__)

def validateConfiguration(data):
    mandatory_fields = ['type', 'images']
    mandatory_fields_image = ['URLImage', 'date']
    if any(x not in list(data.keys()) for x in mandatory_fields):
        raise ValueError("One or more mandatory field(s) is missing")
    s = f = None

    images = data['images']
    for image in images:
        print(image.keys())
        if any(x not in list(image.keys()) for x in mandatory_fields_image):
            raise ValueError("One or more mandatory field(s) is missing from image")
    try:
        s = datetime.strptime(data['start_date'], "%Y-%m-%d")
        f = datetime.strptime(data['final_date'], "%Y-%m-%d")
    except ValueError:
        raise ValueError("Incorrect date format")
    if s > f:
        raise ValueError('Invalid crawling window')
    if 'n_results' in data and not data['n_results'].isdigit():
        raise ValueError('Invalid number of results')

@app.route('/group3output/')
def serve_files():
    file_dir = os.getcwd()+"/files/output_folder"
    files = os.listdir(file_dir)
    elements = []
    for filename in files:
        f = open(file_dir + '/' + filename)
        contents = json.load(f)
        elements.append(contents)
    return json.dumps(elements)

@app.route('/processing/start', methods=['POST'])
def start_processing():
    json_files = request.files.getlist('files')
    # 'files' is the name of the file input field in the form

    # Process each JSON file
    for json_file in json_files:
        # Read the contents of the file
        json_data = json_file.read()

        # Parse the JSON data
        try:
            json_payload = json.loads(json_data)
            #validateConfiguration(json_payload)
            process_image(json_payload)
        except ValueError:
            raise ValueError('Invalid JSON format')
        # Continue processing the JSON payload as needed

    return "done"