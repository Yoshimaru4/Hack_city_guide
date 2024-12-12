import runpod
import base64
from dotenv import load_dotenv
import os
# Load API key from .env file
load_dotenv()
runpod.api_key = os.environ.get("runpod_api_key")

# Initialize the endpoint with your Endpoint ID
endpoint = runpod.Endpoint("wfcsmz2vwv9ndk")  # Replace with your actual endpoint ID

# Encode the audio file in base64
def encode_audio_to_base64(file_path):
    with open(file_path, "rb") as audio_file:
        return base64.b64encode(audio_file.read()).decode('utf-8')

# Convert the local audio file to base64 format
audio_base64 = encode_audio_to_base64("4.wav")

# Define the payload for the model input
model_input = {
    "input": {
        "audio_base64": audio_base64  # Send the audio file as a base64-encoded string
    }
}

# Run the endpoint
run_request = endpoint.run(model_input)

# Poll the status of the request
while run_request.status() not in ['COMPLETED', 'FAILED']:
    print("Status of the request:", run_request.status())

# Get the output of the endpoint run request, blocking until the run is complete
output = run_request.output()
if run_request.status() == 'COMPLETED':
    print("Transcription Output:", output.get('transcription'))
else:
    print("Request failed.")
