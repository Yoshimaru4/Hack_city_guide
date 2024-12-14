import os
import base64
import time
import io
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import together
from dotenv import load_dotenv, find_dotenv
import runpod
import numpy as np
from scipy.io.wavfile import write

load_dotenv(find_dotenv())

# ---------------------------
# Конфигурация Together API
# ---------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = 'meta-llama/Llama-Vision-Free'
llm_client = together.Together(api_key=TOGETHER_API_KEY)

# ---------------------------
# Конфигурация RunPod для TTS и STT
# ---------------------------
runpod.api_key = os.environ.get("runpod_api_key")
stt_endpoint = runpod.Endpoint("wfcsmz2vwv9ndk")  # STT Endpoint
tts_endpoint = runpod.Endpoint("597fbvyobnomsd")  # TTS Endpoint

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
    wav_array = np.array(wav_data, dtype=np.float32)
    wav_array = np.int16(wav_array * 32767)
    write(file_name, sample_rate, wav_array)
    print(f"Saved WAV file: {file_name}")

def transcribe_audio_data(audio_bytes):
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    model_input = {
        "input": {
            "audio_base64": audio_base64
        }
    }
    run_request = stt_endpoint.run(model_input)

    while run_request.status() not in ['COMPLETED', 'FAILED']:
        time.sleep(1)

    if run_request.status() == 'COMPLETED':
        output = run_request.output()
        transcription = output.get('transcription', None)
        return transcription
    else:
        return None

def generate_tts_response(text, speaker="Aaron Dreschner", language="ru"):
    def chunk_text(text, max_words=50):
        """
        Splits text into chunks with a maximum number of words.
        """
        words = text.split()
        for i in range(0, len(words), max_words):
            yield ' '.join(words[i:i + max_words])

    all_wav_data = []

    for chunk in chunk_text(text):
        model_input = {
            "input": {
                "text": chunk,
                "speaker": speaker,
                "language": language
            }
        }
        run_request = tts_endpoint.run(model_input)

        while run_request.status() not in ['COMPLETED', 'FAILED']:
            time.sleep(1)

        if run_request.status() == 'COMPLETED':
            output = run_request.output()
            wav_data = output.get('wav', None)
            if wav_data:
                all_wav_data.extend(wav_data)
        else:
            return None

    return all_wav_data

def play_and_save_tts_response(text, file_name, speaker="Aaron Dreschner", language="ru"):
    wav_data = generate_tts_response(text, speaker, language)
    if wav_data:
        save_wav(wav_data, file_name)
        st.audio(file_name, format="audio/wav")
    else:
        st.error("Ошибка при генерации голосового ответа.")

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

# ---------------------------
# Конфигурация страницы Streamlit
# ---------------------------
def configure_page():
    st.set_page_config(
        page_title="🏠 Поиск ближайших объектов в Москве",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def display_title():
    st.title("🏠 Поиск ближайших объектов в Москве")
    st.markdown("""
        Введите ваше местоположение и укажите радиус поиска, чтобы найти ближайшие дома, места или памятники в указанном диапазоне.
    """)

def user_inputs_sidebar():
    st.sidebar.header("🔍 Ввод данных")
    use_map = st.sidebar.checkbox("🗺️ Выбрать местоположение на карте", value=False)
    distance_input = st.sidebar.number_input(
        "📏 Радиус поиска (метры):",
        min_value=100,
        max_value=5000,
        value=100,
        step=100
    )
    search_button = st.sidebar.button("🔎 Найти ближайшие объекты")
    return  use_map, distance_input, search_button

@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Набор данных не найден по пути {filepath}. Пожалуйста, убедитесь, что файл существует.")
        return pd.DataFrame()
    df = pd.read_csv(filepath, encoding='utf-8')
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

def initialize_session_state():
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'dialog_history' not in st.session_state:
        st.session_state.dialog_history = {}
    if 'selected_item_id' not in st.session_state:
        st.session_state.selected_item_id = None
    if 'last_audio_bytes' not in st.session_state:
        st.session_state.last_audio_bytes = b''

@st.cache_data
def geocode_address(address, geolocator):
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.error(f"Ошибка геокодирования адреса: {e}")
    return (None, None)

def calculate_distances(df, user_location):
    df['Distance_m'] = df.apply(
        lambda row: geodesic(user_location, (row['Latitude'], row['Longitude'])).meters,
        axis=1
    )
    return df

def display_search_results(filtered_df, distance_input):
    st.subheader(f"📌 Объекты в радиусе {distance_input} метров от вашего местоположения:")
    st.write(f"**Всего найдено объектов:** {filtered_df.shape[0]}")
    
    for index, row in filtered_df.iterrows():
        cols = st.columns([1, 3, 2, 2, 1, 1])
        cols[0].write(row['ID'])
        cols[1].write(row['Category'])
        cols[2].write(row['Name'])
        cols[3].write(row['Geo'])
        cols[4].write(f"{row['Distance_m']:.2f} м")
        if cols[5].button("💬 Чат", key=f"chat_button_{row['ID']}"):
            st.session_state.selected_item_id = row['ID']
    
    display_map(filtered_df)

def display_map(filtered_df):
    st.subheader("🗺️ Карта ближайших объектов:")
    folium_map = folium.Map(location=st.session_state.user_location, zoom_start=15)

    folium.Marker(
        location=st.session_state.user_location,
        popup="📍 Ваше местоположение",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(folium_map)

    marker_cluster = MarkerCluster().add_to(folium_map)

    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            popup=f"{row['Name']} (ID: {row['ID']})<br>Расстояние: {row['Distance_m']:.2f} метров",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    st_folium(folium_map, width=700, height=500)

def chat(info, dialog_history):
    if not llm_client:
        st.error("Клиент LLM не инициализирован. Пожалуйста, проверьте конфигурацию Together API.")
        return "Ошибка: Клиент LLM не инициализирован."
    
    messages = [
        {"role": "user", "content": f'Веди себя как полезный и шутливый городской гид, который рассказывает о достопримечательности. Вот информация о ней: {info} \n Говори только про эту достопримечательность, а задавай новые вопросы. Когда расскажешь все, что знаешь, скажи об этом пользователю. После твоего следующего ответа начнется диалог с пользователем. Ты понял?'}
    ]
    messages += [{"role": "assistant", "content": 'Да, я все понял! Начнем экскурсию?'}]
    if dialog_history:
        formatted_history = []
        for msg in dialog_history:
            formatted_history.append({"role": msg['role'], "content": msg['content']})
        messages += formatted_history

    try:
        response = llm_client.chat.completions.create(model=MODEL_NAME, messages=messages)
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Ошибка при завершении чата: {e}")
        return "Ошибка: Не удалось получить ответ."

def display_chat(item_id, use_voice):
    st.markdown(f"### 💬 Чат для объекта ID: {item_id}")
    df = st.session_state.search_results
    info = df.loc[df['ID'] == item_id, 'Info'].values[0]
    if item_id not in st.session_state.dialog_history:
        st.session_state.dialog_history[item_id] = []

    # Display existing messages
    for msg in st.session_state.dialog_history[item_id]:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg["content"])
        elif msg['role'] == 'assistant':
            st.chat_message("assistant").write(msg["content"])

    # Text-based input
    prompt = st.chat_input("Вы:")
    if prompt:
        st.session_state.dialog_history[item_id].append({'role': 'user', 'content': prompt})
        st.chat_message("user").write(prompt)

        df = st.session_state.search_results
        info = df.loc[df['ID'] == item_id, 'Info'].values[0]

        with st.spinner("🤖 Отвечаю..."):
            response = chat(info, st.session_state.dialog_history[item_id])

        st.session_state.dialog_history[item_id].append({'role': 'assistant', 'content': response})
        st.chat_message("assistant").write(response)

        if use_voice:
            with st.spinner("🎙️ Генерация голосового ответа..."):
                file_name = f"response_{item_id}.wav"
                play_and_save_tts_response(response, file_name)

    st.divider()
    st.markdown("#### 🎤 Запись голоса с микрофона")

    # Show audio input widget (available since Streamlit 1.41.0)
    audio_value = st.audio_input("Record a voice message")

    # If user just finished recording and we have new audio
    if audio_value:
        wav_bytes = audio_value.getvalue()
        if wav_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = wav_bytes

            with st.spinner("Транскрипция аудио..."):
                transcription = transcribe_audio_data(wav_bytes)

            if transcription:
                st.session_state.dialog_history[item_id].append({'role': 'user', 'content': transcription})
                st.chat_message("user").write(transcription)

                with st.spinner("🤖 Отвечаю..."):
                    response = chat(info, st.session_state.dialog_history[item_id])

                st.session_state.dialog_history[item_id].append({'role': 'assistant', 'content': response})
                st.chat_message("assistant").write(response)

                if use_voice:
                    with st.spinner("🎙️ Генерация голосового ответа..."):
                        file_name = f"response_{item_id}.wav"
                        play_and_save_tts_response(response, file_name)
            else:
                st.error("Не удалось получить транскрипт из ответа.")

def main():
    configure_page()
    display_title()
    initialize_session_state()
    use_map, distance_input, search_button = user_inputs_sidebar()

    # Add a checkbox for enabling/disabling voice mode
    use_voice = st.sidebar.checkbox("🎙️ Включить голосовой режим", value=False)

    data_path = os.path.join("datasets", "details_filtered.csv")
    df = load_data(data_path)

    if df.empty:
        st.stop()

    geolocator = Nominatim(user_agent="streamlit_app")

    if use_map:
        m = folium.Map(location=[55.7558, 37.6173], zoom_start=10)
        folium.LatLngPopup().add_to(m)
        st.sidebar.markdown("### 📍 Нажмите на карту, чтобы выбрать ваше местоположение:")
        with st.sidebar:
            selected_location = st_folium(m, width=280, height=400)
        if selected_location and 'last_clicked' in selected_location and selected_location['last_clicked']:
            user_lat = selected_location['last_clicked']['lat']
            user_lon = selected_location['last_clicked']['lng']
            st.session_state.user_location = (user_lat, user_lon)
        elif use_map and st.session_state.user_location is None:
            st.session_state.user_location = None

    if search_button:
        with st.spinner("🔄 Обработка..."):
            if use_map:
                if st.session_state.user_location is None:
                    st.error("❗ Пожалуйста, нажмите на карту, чтобы выбрать ваше местоположение.")
                    st.session_state.search_results = None
                else:
                    user_lat, user_lon = st.session_state.user_location

            if st.session_state.user_location:
                user_location = st.session_state.user_location
                df_with_dist = calculate_distances(df, user_location)
                filtered_df = df_with_dist[df_with_dist['Distance_m'] <= distance_input].copy()
                if filtered_df.empty:
                    st.warning("🚫 В указанном радиусе объектов не найдено.")
                    st.session_state.search_results = None
                else:
                    top5_df = filtered_df.nsmallest(5, 'Distance_m').reset_index(drop=True)
                    st.session_state.search_results = top5_df

    if st.session_state.search_results is not None:
        display_search_results(st.session_state.search_results, distance_input)

    if st.session_state.selected_item_id is not None:
        st.header("💬 Интерфейс Чата")
        display_chat(st.session_state.selected_item_id, use_voice)

if __name__ == "__main__":
    main()
