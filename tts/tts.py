import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

text = """На этапе комплексной экспертизы проекта, в том числе в процессе финансово-экономической экспертизы, Заявителю следует корректировать данные
финансовой модели проекта и связанные с ней документы (бизнес-план,
смета, прочие документы), в том числе с учетом замечаний экспертов.
 Во всех случаях внесения изменений в документы Заявитель обязан обеспечить приведение всех документов проекта в соответствие друг другу.
 Данные финансовой модели не должны противоречить данным, содержащимся в других документах по проекту (в том числе резюме, смете, бизнесплане, календарном плане).
 Основные финансовые показатели по проекту и компании с учетом проекта
должны быть отражены на отдельном листе "Выводы".
 Перед направлением документов на комплексную экспертизу при первичном
рассмотрении, а также при каждой корректировке финансовой модели в процессе экспертизы необходимо проверить соответствие следующих показателей:"""
speaker = "Aaron Dreschner"
language = 'ru'

wav = tts.tts(
        text=text,
        speaker=speaker,
        language=language,
    )
import numpy as np
from scipy.io.wavfile import write

def save_wav(wav_data, file_name):
    """
    Save a list of floating-point values as a WAV file.

    Parameters:
    - wav_data (list or ndarray): The audio data, a list or array of floats.
    - sample_rate (int): The sample rate (e.g., 44100 for 44.1 kHz).
    - file_name (str): The name of the output WAV file.

    Returns:
    - None
    """
    sample_rate = 22050
    # Ensure wav_data is a NumPy array
    wav_array = np.array(wav_data, dtype=np.float32)

    # Normalize to 16-bit PCM range (-32768 to 32767) if needed
    wav_array = np.int16(wav_array * 32767)

    # Save the WAV file
    write(file_name, sample_rate, wav_array)
    print(f"Saved WAV file: {file_name}")

save_wav(wav,22050 , 'test.wav')