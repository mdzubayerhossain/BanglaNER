import os
import json
import requests
import time
from flask import Flask, request, jsonify
from inference import BanglaNER
from utils.model_downloading import download_file

# Get the current working directory
root_dir = os.getcwd()

# Define the model directory
model_dir = os.path.join(root_dir, "models")

# Print a message indicating the model is being downloaded
print("Downloading model ......")

# Download the model file
model_dir = download_file(model_dir)

# Set debug mode to True
DEBUG = True

# Initialize the Flask application
app = Flask(__name__)

# Initialize the BanglaNER model
bner = BanglaNER(model_dir)

# Define the NER endpoint
@app.route('/ner', methods=['POST'])
def bangla_ner():
    """
    Endpoint for Bangla NER.
    """
    # Record the start time
    st = time.time()

    # Get the JSON data from the request
    data = request.get_json()
    sender = data.get('sender_id', '')
    text = data.get('text', '')

    # Print the request details
    print("request : ", request)
    print(f"sender : {sender}")
    print(f"text : {text}")

    # Get the prediction from the model
    prediction = bner.prediction(text)

    # Prepare the response
    response = {
        "sender_id" : sender,
        "body"      : prediction,
        "status"    : 200
    }

    # Return the response as a JSON string
    return json.dumps(response, ensure_ascii=False, indent=2)

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8008)
