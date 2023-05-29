import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from process_image import process_image
from process_image import update_timestamp
from datetime import datetime

from tensorflow import keras

app = Flask(__name__)

filter_model = None
classifier_models = []

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

@app.route('/group3output/', methods=['POST'] )
def serve_files():
    """ The purpose of this code is to generate a json response, containing a vector of documents
        in which each document is a file in the output folder
    """
    json_files = request.files.getlist('files')
    # 'files' is the name of the file input field in the form
    new_events = []
    new_timestamps = []
    # Process each JSON file
    for json_file in json_files:
        # Read the contents of the file
        json_data = json_file.read()

        # Parse the JSON data
        try:
            json_payload = json.loads(json_data)
            #validateConfiguration(json_payload)
            stuff, timestamp = process_image(json_payload, filter_model, classifier_models)
            new_events.append(stuff)
            new_timestamps.append(timestamp)
        except ValueError:
            raise ValueError('Invalid JSON format')
        # Continue processing the JSON payload as needed

    update_timestamp(max(new_timestamps))
    return json.dumps(new_events)

@app.route('/visualize_all_events/')
def start_processing():
    """ The purpose of this code is to handle POST requests to '/processing/start' that contain
        uploaded files. It reads the contents of each JSON file, parses the JSON data, and passes
        it to the process_image() function for further processing. If any of the uploaded files do
        not contain valid JSON, a ValueError is raised. Finally, the route returns "done" as the
        response to indicate that the processing is complete.
    """
    file_dir = os.getcwd() + "/files/output_folder"
    files = os.listdir(file_dir)
    elements = []
    for filename in files:
        f = open(file_dir + '/' + filename)
        contents = json.load(f)
        elements.append(contents)
    return json.dumps(elements)

def load_models():
    global filter_model, classifier_models

    # Load the filter model
    filter_dir = "models/filter/tl_binary.h5"
    filter_model = keras.models.load_model(filter_dir)
    print("Loading model: " + filter_dir)

    # Load the classifier models
    classifier_dir = "models/ensamble1/"
    for file_name in os.listdir(classifier_dir):
        model_path = os.path.join(classifier_dir, file_name)
        print("Loading model: " + model_path)
        classifier_model = keras.models.load_model(model_path)
        classifier_models.append(classifier_model)


# Call the load_models function to load the models when the Flask app starts
load_models()

# debugging reasons
#if __name__ == '__sample__':
    # to run on server
    # app.run(host='3.69.174.135', port=3333)

    # to run on localhost
    #app.run(host='localhost', port=5000)
