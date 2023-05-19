import os
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime
import json
from process_image import process_image

app = Flask(__name__)

def validateConfiguration(data):
    mandatory_fields = ['keywords','start_date','final_date']   #our keys
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
    file_dir = os.getcwd()+"/files/output_folder"
    files = os.listdir(file_dir)
    response = ""
    for filename in files:
        response += f"<a href='/group3output/{filename}'>{filename}</a><br>"
    return response

@app.route('/group3output/<path:filename>')
def serve_file(filename):
    return send_from_directory(os.getcwd()+"/files/output_folder", filename)

@app.route('/crowd4sdg/start', methods=['POST'])
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
            process_image(json_payload)
        except ValueError:
            raise ValueError('Invalid JSON format')
        # Continue processing the JSON payload as needed

    return "done"

if __name__ == '__sample__':
    app.run(host='3.69.174.135', port=3333)