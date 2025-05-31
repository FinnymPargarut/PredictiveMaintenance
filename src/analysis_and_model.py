import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve


def load_data_from_ucirepo():
    dataset = fetch_ucirepo(id=601)
    data = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    st.write("Названия столбцов в датасете:", data.columns.tolist())  # Для отладки
    return data


def preprocess_data(data):
    # Проверяем и удаляем только существующие столбцы
    columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
    existing_columns = [col for col in columns_to_drop if col in data.columns]
    data = data.drop(columns=existing_columns)

    # Преобразование категориальной переменной 'Type'
    if 'Type' in data.columns:
        data['Type'] = LabelEncoder().fit_transform(data['Type'])

    # Список числовых признаков, соответствующий реальным столбцам датасета
    numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque', 'Tool wear']
    existing_numerical = [col for col in numerical_features if col in data.columns]

    # Проверка, что есть числовые столбцы для масштабирования
    scaler = StandardScaler()
    if existing_numerical:
        data[existing_numerical] = scaler.fit_transform(data[existing_numerical])
    else:
        st.warning("Числовые признаки для масштабирования не найдены.")

    return data, scaler


def split_data(data):
    X = data.drop(columns=['Machine failure'])  # Целевая переменная
    y = data['Machine failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, conf_matrix, class_report, roc_auc, y_pred_proba


def analysis_and_model_page():
    st.title("Анализ данных и модель")

    # Загрузка данных
    data_source = st.radio("Выберите источник данных", ["Загрузить из файла", "Загрузить с ucimlrepo"])
    data = None
    if data_source == "Загрузить из файла":
        uploaded_file = st.file_uploader("Загрузите датасет (CSV)", type="csv")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
        else:
            st.warning("Пожалуйста, загрузите файл.")
            return
    else:
        data = load_data_from_ucirepo()
        st.write("Названия столбцов в датасете:", data.columns.tolist())

    if data is not None:
        # Предобработка данных
        data, scaler = preprocess_data(data)
        X_train, X_test, y_train, y_test = split_data(data)
        model = train_model(X_train, y_train)
        accuracy, conf_matrix, class_report, roc_auc, y_pred_proba = evaluate_model(model, X_test, y_test)

        # Визуализация результатов
        st.header("Результаты обучения модели")
        st.write(f"Accuracy: {accuracy:.2f}")
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
        st.pyplot(fig)
        st.subheader("Classification Report")
        st.text(class_report)
        st.subheader("ROC-AUC")
        st.write(f"ROC-AUC: {roc_auc:.2f}")
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label=f"Модель (AUC = {roc_auc:.2f})")
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC-кривая')
        ax.legend()
        st.pyplot(fig)

        # Интерфейс для предсказания
        st.header("Предсказание по новым данным")
        with st.form("prediction_form"):
            type_input = st.selectbox("Type", ["L", "M", "H"])
            air_temp = st.number_input("Air temperature")
            process_temp = st.number_input("Process temperature")
            rotational_speed = st.number_input("Rotational speed")
            torque = st.number_input("Torque")
            tool_wear = st.number_input("Tool wear")
            submit_button = st.form_submit_button("Предсказать")
            if submit_button:
                type_encoded = LabelEncoder().fit_transform([type_input])[0]
                input_data = pd.DataFrame({
                    'Type': [type_encoded],
                    'Air temperature': [air_temp],
                    'Process temperature': [process_temp],
                    'Rotational speed': [rotational_speed],
                    'Torque': [torque],
                    'Tool wear': [tool_wear]
                })
                numerical_features = ['Air temperature', 'Process temperature', 'Rotational speed', 'Torque',
                                      'Tool wear']
                existing_numerical = [col for col in numerical_features if col in input_data.columns]
                if existing_numerical:
                    input_data[existing_numerical] = scaler.transform(input_data[existing_numerical])
                prediction = model.predict(input_data)
                prediction_proba = model.predict_proba(input_data)[:, 1]
                st.write(f"Предсказание: {'Отказ' if prediction[0] == 1 else 'Нет отказа'}")
                st.write(f"Вероятность отказа: {prediction_proba[0]:.2f}")