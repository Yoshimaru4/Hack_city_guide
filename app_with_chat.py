# app.py

import os
import pandas as pd
import streamlit as st
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
from streamlit_folium import st_folium
import folium
from folium.plugins import MarkerCluster
import time  # Используется в синхронной функции чата
import together
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# ---------------------------
# Конфигурация Together API
# ---------------------------
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
MODEL_NAME = 'meta-llama/Llama-Vision-Free'
llm_client = together.Together(api_key=TOGETHER_API_KEY)

# ---------------------------
# Конфигурация страницы Streamlit
# ---------------------------
def configure_page():
    st.set_page_config(
        page_title="🏠 Поиск ближайших объектов в Москве",
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ---------------------------
# Заголовок и описание
# ---------------------------
def display_title():
    st.title("🏠 Поиск ближайших объектов в Москве")
    st.markdown("""
        Введите ваше местоположение и укажите радиус поиска, чтобы найти ближайшие дома, места или памятники в указанном диапазоне.
    """)

# ---------------------------
# Боковая панель для ввода данных пользователем
# ---------------------------
def user_inputs_sidebar():
    st.sidebar.header("🔍 Ввод данных")
    location_input = st.sidebar.text_input(
        "📍 Введите ваш адрес в Москве:",
        placeholder="например, Красная площадь, Москва"
    )
    use_map = st.sidebar.checkbox("🗺️ Выбрать местоположение на карте", value=False)
    distance_input = st.sidebar.number_input(
        "📏 Радиус поиска (метры):",
        min_value=100,
        max_value=5000,
        value=100,
        step=100
    )
    search_button = st.sidebar.button("🔎 Найти ближайшие объекты")
    return location_input, use_map, distance_input, search_button

# ---------------------------
# Функция загрузки данных
# ---------------------------
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Набор данных не найден по пути {filepath}. Пожалуйста, убедитесь, что файл существует.")
        return pd.DataFrame()
    df = pd.read_csv(filepath, encoding='utf-8')
    # Присвоение уникального ID на основе индекса DataFrame
    df.reset_index(inplace=True)
    df.rename(columns={'index': 'ID'}, inplace=True)
    return df

# ---------------------------
# Инициализация состояния сессии
# ---------------------------
def initialize_session_state():
    if 'user_location' not in st.session_state:
        st.session_state.user_location = None  # Кортеж (широта, долгота)
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None  # DataFrame с результатами поиска
    if 'dialog_history' not in st.session_state:
        st.session_state.dialog_history = {}  # Словарь с ID объекта в качестве ключа и списком сообщений
    if 'selected_item_id' not in st.session_state:
        st.session_state.selected_item_id = None  # В данный момент выбранный объект для чата

# ---------------------------
# Геокодирование
# ---------------------------
@st.cache_data
def geocode_address(address, geolocator):
    try:
        location = geolocator.geocode(address)
        if location:
            return (location.latitude, location.longitude)
    except Exception as e:
        st.error(f"Ошибка геокодирования адреса: {e}")
    return (None, None)

# ---------------------------
# Расчет расстояний
# ---------------------------
def calculate_distances(df, user_location):
    df['Distance_m'] = df.apply(
        lambda row: geodesic(user_location, (row['Latitude'], row['Longitude'])).meters,
        axis=1
    )
    return df

# ---------------------------
# Отображение результатов поиска
# ---------------------------
def display_search_results(filtered_df, distance_input):
    st.subheader(f"📌 Объекты в радиусе {distance_input} метров от вашего местоположения:")
    st.write(f"**Всего найдено объектов:** {filtered_df.shape[0]}")
    
    # Отображение в виде таблицы с кнопками "Чат"
    for index, row in filtered_df.iterrows():
        cols = st.columns([1, 3, 2, 2, 1, 1])  # Настройка ширины колонок при необходимости
        cols[0].write(row['ID'])
        cols[1].write(row['Category'])
        cols[2].write(row['Name'])
        cols[3].write(row['Geo'])
        cols[4].write(f"{row['Distance_m']:.2f} м")
        if cols[5].button("💬 Чат", key=f"chat_button_{row['ID']}"):
            st.session_state.selected_item_id = row['ID']
    
    # Отображение на карте
    display_map(filtered_df)

# ---------------------------
# Отображение карты с маркерами
# ---------------------------
def display_map(filtered_df):
    st.subheader("🗺️ Карта ближайших объектов:")
    folium_map = folium.Map(location=st.session_state.user_location, zoom_start=15)

    # Добавление маркера для местоположения пользователя
    folium.Marker(
        location=st.session_state.user_location,
        popup="📍 Ваше местоположение",
        icon=folium.Icon(color='red', icon='user')
    ).add_to(folium_map)

    # Инициализация кластеризации маркеров
    marker_cluster = MarkerCluster().add_to(folium_map)

    # Добавление маркеров для ближайших объектов
    for _, row in filtered_df.iterrows():
        folium.Marker(
            location=(row['Latitude'], row['Longitude']),
            popup=f"{row['Name']} (ID: {row['ID']})<br>Расстояние: {row['Distance_m']:.2f} метров",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(marker_cluster)

    # Отображение карты
    st_folium(folium_map, width=700, height=500)

# ---------------------------
# Синхронная функция чата
# ---------------------------
def chat(info, dialog_history):
    """
    Синхронная функция чата с использованием Together API.
    Замените эту функцию реальной логикой чата, например, вызовами API.
    """
    if not llm_client:
        st.error("Клиент LLM не инициализирован. Пожалуйста, проверьте конфигурацию Together API.")
        return "Ошибка: Клиент LLM не инициализирован."
    
    messages = [
        {"role": "user", "content": f'Веди себя как полезный и шутливый городской гид, который рассказывает о достопримечательности. Вот информация о ней: {info} \n Говори только про эту достопримечательность, а задавай новые вопросы. Когда расскажешь все, что знаешь, скажи об этом пользователю. После твоего следующего ответа начнется диалог с пользователем. Ты понял?'}
    ]
    messages += [{"role": "assistant", "content": 'Да, я все понял! Начнем экскурсию?'}]
    if dialog_history:
        # Преобразование истории диалога в необходимый формат
        formatted_history = []
        for msg in dialog_history:
            formatted_history.append({"role": msg['role'], "content": msg['content']})
        messages += formatted_history

    # Вызов API завершения чата Together
    try:
        response = llm_client.chat.completions.create(model=MODEL_NAME, messages=messages)
        # Предполагается, что структура ответа похожа на OpenAI
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Ошибка при завершении чата: {e}")
        return "Ошибка: Не удалось получить ответ."

# ---------------------------
# Отображение интерфейса чата
# ---------------------------
def display_chat(item_id):
    st.markdown(f"### 💬 Чат для объекта ID: {item_id}")
    
    # Инициализация истории диалога для объекта
    if item_id not in st.session_state.dialog_history:
        st.session_state.dialog_history[item_id] = []
    
    # Отображение истории диалога с использованием современных компонентов Streamlit
    for msg in st.session_state.dialog_history[item_id]:
        if msg['role'] == 'user':
            st.chat_message("user").write(msg["content"])
        elif msg['role'] == 'assistant':
            st.chat_message("assistant").write(msg["content"])
    
    # Создание поля ввода чата
    prompt = st.chat_input("Вы:")
    
    if prompt:
        # Добавление сообщения пользователя в историю диалога
        st.session_state.dialog_history[item_id].append({'role': 'user', 'content': prompt})
        st.chat_message("user").write(prompt)
        
        # Получение информации об объекте для чата
        df = st.session_state.search_results
        info = df.loc[df['ID'] == item_id, 'Info'].values[0]
        
        # Вызов синхронной функции чата
        with st.spinner("🤖 Отвечаю..."):
            response = chat(info, st.session_state.dialog_history[item_id])
        
        # Добавление ответа помощника в историю диалога
        st.session_state.dialog_history[item_id].append({'role': 'assistant', 'content': response})
        st.chat_message("assistant").write(response)

# ---------------------------
# Основная логика приложения
# ---------------------------
def main():
    configure_page()
    display_title()
    initialize_session_state()
    location_input, use_map, distance_input, search_button = user_inputs_sidebar()

    # Загрузка набора данных
    data_path = os.path.join("datasets", "details_filtered.csv")
    df = load_data(data_path)

    if df.empty:
        st.stop()

    geolocator = Nominatim(user_agent="streamlit_app")

    # Обработка выбора местоположения на карте
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
            st.session_state.user_location = None  # Сброс, если нет выбора

    # Обработка нажатия кнопки поиска
    if search_button:
        with st.spinner("🔄 Обработка..."):
            # Определение местоположения пользователя
            if use_map:
                if st.session_state.user_location is None:
                    st.error("❗ Пожалуйста, нажмите на карту, чтобы выбрать ваше местоположение.")
                    st.session_state.search_results = None  # Очистка предыдущих результатов
                else:
                    user_lat, user_lon = st.session_state.user_location
            else:
                if not location_input.strip():
                    st.error("❗ Пожалуйста, введите действительный адрес или выберите местоположение на карте.")
                    st.session_state.search_results = None  # Очистка предыдущих результатов
                else:
                    user_lat, user_lon = geocode_address(location_input, geolocator)
                    if user_lat is None or user_lon is None:
                        st.error("❗ Не удалось геокодировать предоставленный адрес. Пожалуйста, попробуйте другой.")
                        st.session_state.search_results = None  # Очистка предыдущих результатов
                    else:
                        st.session_state.user_location = (user_lat, user_lon)  # Обновление состояния сессии

            # Продолжение только при действительном местоположении пользователя
            if st.session_state.user_location:
                user_location = st.session_state.user_location
                # Расчет расстояний
                df_with_dist = calculate_distances(df, user_location)
                # Фильтрация по расстоянию и выбор 5 ближайших объектов
                filtered_df = df_with_dist[df_with_dist['Distance_m'] <= distance_input].copy()
                if filtered_df.empty:
                    st.warning("🚫 В указанном радиусе объектов не найдено.")
                    st.session_state.search_results = None  # Очистка предыдущих результатов
                else:
                    # Сортировка по расстоянию и выбор 5 ближайших объектов
                    top5_df = filtered_df.nsmallest(5, 'Distance_m').reset_index(drop=True)
                    st.session_state.search_results = top5_df

    # Отображение результатов поиска
    if st.session_state.search_results is not None:
        display_search_results(st.session_state.search_results, distance_input)

    # Отображение интерфейса чата для выбранного объекта
    if st.session_state.selected_item_id is not None:
        st.header("💬 Интерфейс Чата")
        display_chat(st.session_state.selected_item_id)

if __name__ == "__main__":
    main()
