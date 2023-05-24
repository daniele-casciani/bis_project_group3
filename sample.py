import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from process_image import process_image
from datetime import datetime


app = Flask(__name__)

def validateConfiguration(data):
    """ This function is designed to validate the configuration dictionary by checking
        the presence of mandatory fields, the correctness of date formats, the order of dates.
        If any of the validation checks fail, a ValueError is raised with an appropriate error message.
    """
    mandatory_fields = ['type', 'images']
    mandatory_fields_image = ['image_url', 'timeframe']
    if any(x not in list(data.keys()) for x in mandatory_fields):
        raise ValueError("One or more mandatory field(s) is missing")

    images = data['images']
    for image in images:
        if any(x not in list(image.keys()) for x in mandatory_fields_image):
            raise ValueError("One or more mandatory field(s) is missing from image")
        if len(image['timeframe']) != 2:
            raise ValueError("Invalid timeframe")
        if datetime.fromisoformat(image['timeframe'][0]) > datetime.fromisoformat(image['timeframe'][1]):
            raise ValueError("Invalid timeframe")

@app.route('/group3output/')
def serve_files():
    """ The purpose of this code is to generate a json response, containing a vector of documents
        in which each document is a file in the output folder
    """
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
    """ The purpose of this code is to handle POST requests to '/processing/start' that contain
        uploaded files. It reads the contents of each JSON file, parses the JSON data, and passes
        it to the process_image() function for further processing. If any of the uploaded files do
        not contain valid JSON, a ValueError is raised. Finally, the route returns "done" as the
        response to indicate that the processing is complete.
    """
    json_files = request.files.getlist('files')
    # 'files' is the name of the file input field in the form

    # Process each JSON file
    for json_file in json_files:
        # Read the contents of the file
        json_data = json_file.read()

        # Parse the JSON data
        try:
            json_payload = json.loads(json_data)
            validateConfiguration(json_payload)
            process_image(json_payload)
        except ValueError:
            raise ValueError('Invalid JSON format')
        # Continue processing the JSON payload as needed

    return "done"

# debugging reasons
#if __name__ == '__sample__':
    # to run on server
    # app.run(host='3.69.174.135', port=3333)

    # to run on localhost
    #app.run(host='localhost', port=5000)