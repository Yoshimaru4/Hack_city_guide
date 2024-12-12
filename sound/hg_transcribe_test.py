from huggingsound import SpeechRecognitionModel
import torch
import time
model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-russian", device = 'cuda')
audio_paths = ["4.wav"]
tm = time.time()
transcriptions = model.transcribe(audio_paths)

print(transcriptions[0]['transcription'])
print(time.time()-tm)