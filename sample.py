import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from process_image import process_image

app = Flask(__name__)

def validateConfiguration(data):
    """ This function is designed to validate the configuration dictionary by checking 
        the presence of mandatory fields, the correctness of date formats, the order of dates,
        and the validity of the number of results if provided. If any of the validation 
        checks fail, a ValueError is raised with an appropriate error message.
    """
    mandatory_fields = ['keywords','start_date','final_date']   # TODO are just hust these?
    if any(x not in list(data.keys()) for x in mandatory_fields):
        raise ValueError("One or more mandatory field(s) is missing")
    s = f = None
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
    """ The purpose of this code is to generate an HTML response that displays the list 
        of files in the output_folder. Each file is shown as a clickable link, allowing 
        users to access individual files by clicking on the corresponding link.
    """
    file_dir = os.getcwd()+"/files/output_folder"
    files = os.listdir(file_dir)
    response = ""
    for filename in files:
        response += f"<a href='/group3output/{filename}'>{filename}</a><br>"
    return response

@app.route('/group3output/<path:filename>')
def serve_file(filename):
    """ The purpose of this code is to handle requests for specific files in the output_folder. 
        When a request is made to /group3output/<filename>, the corresponding file is located in 
        the output_folder directory, and Flask sends it back to the client as a response.
    """
    return send_from_directory(os.getcwd()+"/files/output_folder", filename)

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
            process_image(json_payload)
        except ValueError:
            raise ValueError('Invalid JSON format')
        # Continue processing the JSON payload as needed

    return "done"

if __name__ == '__sample__':
    app.run(host='3.69.174.135', port=3333)