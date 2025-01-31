# Import necessary libraries
import os
import json
import requests
import time
from flask import Flask, request, jsonify
from inference import BanglaNER  # Import the BanglaNER class from the inference module

# Set debug mode to True
DEBUG = True

# Initialize the Flask application
app = Flask(__name__)

# Define a configuration class to hold the model path
class Config:
    model_path = "./models/bangla_ner_model"  # Path to the Bangla NER model

# Create an instance of the configuration class
cfg = Config()

# Initialize the BanglaNER model with the model path
bner = BanglaNER(cfg.model_path)

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
    sender = data.get('sender_id', '')  # Get the sender ID from the JSON data
    text = data.get('text', '')  # Get the text from the JSON data

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
    app.run(debug=True, host="0.0.0.0", port=8008)  # Run the app in debug mode on all available IP addresses and port 8008
