# model_training.py - Skrypt do trenowania modelu Random Forest
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_prepare_data(file_path='mieszkania_pelny_zbior.csv'):
    """
    Wczytuje dane z pliku CSV i przygotowuje je do modelowania
    
    Args:
        file_path (str): Ścieżka do pliku CSV z danymi
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test) - podzielone dane treningowe i testowe
    """
    # Wczytaj dane
    print("Wczytywanie danych...")
    df = pd.read_csv(file_path)
    
    # Wykonaj czyszczenie danych - zaimportuj funkcję z data_collector.py
    from data_collector import clean_dataset
    print("Czyszczenie danych...")
    clean_df = clean_dataset(df)
    
    # Przygotuj dane do modelowania
    from data_collector import prepare_for_modeling
    print("Przygotowywanie cech...")
    X, y = prepare_for_modeling(clean_df)
    
    # Podziel na zbiór treningowy i testowy
    print("Dzielenie na zbiory treningowy i testowy...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Liczba próbek treningowych: {len(X_train)}")
    print(f"Liczba próbek testowych: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, X.columns

def build_preprocessing_pipeline(X):
    """
    Buduje pipeline do przetwarzania wstępnego danych
    
    Args:
        X (DataFrame): DataFrame z cechami
        
    Returns:
        ColumnTransformer: Preprocesor do przekształcania danych
    """
    # Identyfikuj kolumny numeryczne i kategoryczne
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print("Cechy kategoryczne:", categorical_features)
    print("Cechy numeryczne:", numerical_features)
    
    # Utwórz preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    return preprocessor

def train_random_forest(X_train, y_train, preprocessor, tune_hyperparams=False):
    """
    Trenuje model Random Forest
    
    Args:
        X_train (DataFrame): Cechy treningowe
        y_train (Series): Etykiety treningowe
        preprocessor (ColumnTransformer): Preprocesor do danych
        tune_hyperparams (bool): Czy optymalizować hiperparametry
        
    Returns:
        Pipeline: Wytrenowany model wraz z preprocessingiem
    """
    if tune_hyperparams:
        # Utwórz pipeline z preprocessorem i modelem
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(random_state=42))
        ])
        
        # Zdefiniuj przestrzeń hiperparametrów do przeszukania
        param_grid = {
            'regressor__n_estimators': [50, 100, 200],
            'regressor__max_depth': [None, 10, 20, 30],
            'regressor__min_samples_split': [2, 5, 10],
            'regressor__min_samples_leaf': [1, 2, 4]
        }
        
        # Utwórz GridSearchCV
        print("Optymalizacja hiperparametrów (to może potrwać)...")
        grid_search = GridSearchCV(
            pipeline, param_grid=param_grid, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=-1, verbose=1
        )
        
        # Trenuj model z optymalizacją hiperparametrów
        grid_search.fit(X_train, y_train)
        
        # Wyświetl najlepsze hiperparametry
        print("Najlepsze hiperparametry:")
        print(grid_search.best_params_)
        
        # Pobierz najlepszy model
        best_pipeline = grid_search.best_estimator_
        
        return best_pipeline
    else:
        # Podstawowy model bez optymalizacji
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        print("Trenowanie modelu Random Forest...")
        pipeline.fit(X_train, y_train)
        
        return pipeline

def evaluate_model(model, X_test, y_test):
    """
    Ocenia model na podstawie danych testowych
    
    Args:
        model (Pipeline): Wytrenowany model
        X_test (DataFrame): Cechy testowe
        y_test (Series): Etykiety testowe
        
    Returns:
        dict: Słownik ze statystykami oceny modelu
    """
    print("Ocena modelu na danych testowych...")
    
    # Wykonaj predykcje
    y_pred = model.predict(X_test)
    
    # Oblicz metryki
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Wyświetl metryki
    print(f"Mean Absolute Error (MAE): {mae:.2f} PLN")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} PLN")
    print(f"R² Score: {r2:.4f}")
    
    # Oblicz procentowy błąd względny
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    
    # Przygotuj słownik z metrykami
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }
    
    # Wizualizacja predykcji vs rzeczywiste wartości
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Rzeczywista cena')
    plt.ylabel('Przewidywana cena')
    plt.title('Porównanie: przewidywane vs rzeczywiste ceny mieszkań')
    plt.savefig('model_evaluation_plot.png')
    
    return metrics, y_pred

def analyze_feature_importance(model, feature_names):
    """
    Analizuje ważność cech w modelu
    
    Args:
        model (Pipeline): Wytrenowany model
        feature_names (list): Lista nazw cech
        
    Returns:
        DataFrame: DataFrame z ważnością cech
    """
    # Pobierz model Random Forest z pipeline
    rf_model = model.named_steps['regressor']
    
    # Pobierz ważność cech
    importances = rf_model.feature_importances_
    
    # Pobierz nazwy cech po transformacji (to będzie bardziej skomplikowane)
    # W rzeczywistości trzeba dostosować tę logikę do struktury twojego preprocessora
    
    # Uproszczone podejście - mapuj ważność cech na oryginalne cechy
    # To jest przybliżone, bo transformacja OneHotEncoder zmienia liczbę cech
    
    # Pobierz oryginalne cechy kategoryczne i numeryczne
    preprocessor = model.named_steps['preprocessor']
    categorical_idx = preprocessor.transformers_[1][2]  # Indeksy cech kategorycznych
    numerical_idx = preprocessor.transformers_[0][2]    # Indeksy cech numerycznych
    
    categorical_features = [feature_names[i] for i in categorical_idx]
    numerical_features = [feature_names[i] for i in numerical_idx]
    
    # Przypisz ważność cech
    # Uproszczenie: przypisz średnią ważność dla każdej cechy kategorycznej
    feature_importance = {}
    
    # Dla kategorycznych, oblicz średnią ważność wszystkich one-hot encoded cech
    cat_importances = []
    
    # Utwórz słownik z ważnością cech
    # To jest uproszczone - w praktyce trzeba dostosować do konkretnej implementacji
    for i, feature in enumerate(feature_names):
        # Uproszczone przypisanie ważności
        if i < len(importances):
            feature_importance[feature] = importances[i] * 100
        else:
            # Fallback dla cech, które wyszły poza zakres (z powodu transformacji)
            feature_importance[feature] = 0
    
    # Posortuj cechy według ważności
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Utwórz DataFrame z ważnością cech
    importance_df = pd.DataFrame(sorted_features, columns=['Feature', 'Importance'])
    importance_df['Importance'] = importance_df['Importance'].round(2)
    
    # Wyświetl ważność cech
    print("\nWażność cech:")
    for feature, importance in sorted_features[:10]:  # Topowe 10 cech
        print(f"  - {feature}: {importance:.2f}%")
    
    # Wizualizacja ważności cech
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(10)
    sns.barplot(x='Importance', y='Feature', data=top_features)
    plt.title('Top 10 najważniejszych cech')
    plt.savefig('feature_importance_plot.png')
    
    return importance_df

def save_model(model, file_path='model.pkl'):
    """
    Zapisuje model do pliku
    
    Args:
        model (Pipeline): Wytrenowany model
        file_path (str): Ścieżka do pliku, w którym zostanie zapisany model
    """
    with open(file_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model zapisany do pliku: {file_path}")

def main():
    """
    Główna funkcja trenująca model Random Forest
    """
    # Wczytaj i przygotuj dane
    X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
    
    # Zbuduj preprocessor
    preprocessor = build_preprocessing_pipeline(X_train)
    
    # Trenuj model
    # Ustawienie tune_hyperparams=True włącza optymalizację hiperparametrów (trwa dłużej)
    model = train_random_forest(X_train, y_train, preprocessor, tune_hyperparams=False)
    
    # Oceń model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Analizuj ważność cech
    importance_df = analyze_feature_importance(model, feature_names)
    
    # Zapisz model
    save_model(model)
    
    print("\nTrening modelu zakończony!")
    
    # Opcjonalnie: Zapisz wyniki oceny
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    })
    
    results.to_csv('model_predictions.csv', index=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    
    # Zapisz metryki do pliku
    with open('model_metrics.txt', 'w') as f:
        f.write(f"Mean Absolute Error (MAE): {metrics['mae']:.2f} PLN\n")
        f.write(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.2f} PLN\n")
        f.write(f"R² Score: {metrics['r2']:.4f}\n")
        f.write(f"Mean Absolute Percentage Error (MAPE): {metrics['mape']:.2f}%\n")

if __name__ == "__main__":
    main()