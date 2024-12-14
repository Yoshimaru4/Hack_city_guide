
from dotenv import load_dotenv, find_dotenv
import runpod
import os
import time
import numpy as np
from scipy.io.wavfile import write

load_dotenv(find_dotenv())
runpod.api_key = os.environ.get("runpod_api_key")
endpoint = runpod.Endpoint("597fbvyobnomsd")  # Замените на свой реальный Endpoint ID

model_input = {
  "input": {
    "text":"""На этапе комплексной экспертизы проекта, в том числе в процессе финансово-экономической экспертизы, Заявителю следует корректировать данные
финансовой модели проекта и связанные с ней документы (бизнес-план,
смета, прочие документы), в том числе с учетом замечаний экспертов.""",
    "speaker": "Aaron Dreschner",
    "language": "ru"
  }
}

run_request = endpoint.run(model_input)
while run_request.status() not in ['COMPLETED', 'FAILED']:
    time.sleep(1)
if run_request.status() == 'COMPLETED':
    output = run_request.output()
    wav = output.get('wav', None)

def save_wav(wav_data, sample_rate, file_name):
    """
    Save a list of floating-point values as a WAV file.

    Parameters:
    - wav_data (list or ndarray): The audio data, a list or array of floats.
    - sample_rate (int): The sample rate (e.g., 44100 for 44.1 kHz).
    - file_name (str): The name of the output WAV file.

    Returns:
    - None
    """
    # Ensure wav_data is a NumPy array
    wav_array = np.array(wav_data, dtype=np.float32)

    # Normalize to 16-bit PCM range (-32768 to 32767) if needed
    wav_array = np.int16(wav_array * 32767)

    # Save the WAV file
    write(file_name, sample_rate, wav_array)
    print(f"Saved WAV file: {file_name}")

save_wav(wav,22050 , 'test.wav')