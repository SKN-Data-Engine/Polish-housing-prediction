# backend.py - Główny plik aplikacji Flask
from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ----- Miejsce na załadowanie modelu -----
# Jeśli masz już wytrenowany model możesz go załadować za pomocą pickle:
# model = pickle.load(open('model.pkl', 'rb'))
# 
# Jeśli nie, to poniżej jest struktura do trenowania modelu

# ----- Funkcje pomocnicze -----

def load_data():
    """
    Funkcja do wczytywania danych o mieszkaniach.
    Tutaj możesz zaimplementować wczytywanie danych z CSV lub bazy danych.
    """
    # TODO: Zastąp to rzeczywistym wczytywaniem danych
    # Przykładowe dane (wymyślone) - zastąp prawdziwymi danymi
    data = {
        'city': ['warszawa', 'warszawa', 'krakow', 'wroclaw'],
        'district': ['mokotow', 'srodmiescie', 'stare_miasto', 'krzyki'],
        'area': [65, 48, 72, 55],
        'rooms': [3, 2, 3, 2],
        'floor': [2, 4, 1, 3],
        'buildYear': [2010, 1970, 2005, 2015],
        'condition': ['dobry', 'do_remontu', 'bardzo_dobry', 'nowe'],
        'type': ['blok', 'kamienica', 'apartamentowiec', 'blok'],
        'balcony': [1, 0, 1, 1],
        'parking': [1, 0, 1, 0],
        'elevator': [1, 0, 1, 1],
        'price': [750000, 550000, 850000, 600000]
    }
    return pd.DataFrame(data)

def train_model():
    """
    Funkcja do trenowania modelu Random Forest na danych o mieszkaniach.
    """
    # Wczytaj dane
    df = load_data()
    
    # Podziel na cechy i etykiety
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Podziel na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Zdefiniuj kolumny kategoryczne i numeryczne
    categorical_features = ['city', 'district', 'condition', 'type']
    numerical_features = ['area', 'rooms', 'floor', 'buildYear', 'balcony', 'parking', 'elevator']
    
    # Preprocessor do przekształcania danych
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    # Stwórz potok przetwarzania
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    # Trenuj model
    model.fit(X_train, y_train)
    
    # Opcjonalnie: Zapisz model
    # pickle.dump(model, open('model.pkl', 'wb'))
    
    return model

# Załaduj lub trenuj model przy starcie aplikacji
model = train_model()

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint do przewidywania ceny mieszkania na podstawie podanych parametrów.
    """
    # Pobierz dane z żądania
    data = request.json
    
    # Utwórz DataFrame z danymi
    input_data = pd.DataFrame([data])
    
    # Wykonaj predykcję
    predicted_price = model.predict(input_data)[0]
    
    # Oblicz ważność cech (dla Random Forest możemy to zrobić)
    # Uwaga: to jest uproszczona implementacja, ponieważ potrzebujemy dostępu do modelu wewnątrz pipelines
    feature_importance = {}
    rf_model = model.named_steps['regressor']
    preprocessor = model.named_steps['preprocessor']
    
    # Pobierz nazwy cech po transformacji
    # To jest przykładowe - trzeba dostosować do konkretnej implementacji
    all_features = []
    numerical_features = ['area', 'rooms', 'floor', 'buildYear', 'balcony', 'parking', 'elevator']
    all_features.extend(numerical_features)
    
    # Uproszczone obliczenie ważności cech - w rzeczywistej implementacji 
    # musisz to dostosować do struktury twojego pipeline
    importances = rf_model.feature_importances_
    
    # W rzeczywistej implementacji ta sekcja będzie bardziej złożona
    # Tu używamy przykładowych danych
    feature_importance = {
        'area': 35.2,
        'city': 18.7,
        'district': 15.3,
        'buildYear': 10.5,
        'condition': 8.1,
        'rooms': 5.3,
        'type': 3.2,
        'floor': 2.1,
        'balcony': 0.8,
        'parking': 0.5,
        'elevator': 0.3
    }
    
    # Przygotuj odpowiedź
    response = {
        'price': round(predicted_price, 2),
        'feature_importance': feature_importance
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)