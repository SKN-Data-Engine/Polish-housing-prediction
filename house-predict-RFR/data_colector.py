# data_collector.py - Skrypt do zbierania danych o mieszkaniach z Otodom
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re

def scrape_otodom(pages=10, city="warszawa"):
    """
    Funkcja do scrapowania danych o mieszkaniach z serwisu Otodom.
    
    Args:
        pages (int): Liczba stron do przeszukania
        city (str): Miasto, którego dotyczą ogłoszenia
        
    Returns:
        DataFrame: DataFrame z danymi o mieszkaniach
    """
    base_url = f"https://www.otodom.pl/pl/oferty/sprzedaz/mieszkanie/{city}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    all_apartments = []
    
    for page in range(1, pages + 1):
        url = f"{base_url}?page={page}"
        print(f"Scraping page {page}...")
        
        try:
            response = requests.get(url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Znajdź wszystkie ogłoszenia na stronie
            # Uwaga: Selektory CSS mogą się zmienić, trzeba je dostosować
            apartments = soup.select('div.css-1okcwbv')
            
            for apt in apartments:
                try:
                    # Wyciągnij dane z ogłoszenia
                    # Uwaga: To są przykładowe selektory, trzeba je dostosować
                    price_elem = apt.select_one('p.css-10b0gli')
                    area_elem = apt.select_one('span.css-1uvpk0b:contains("m²")')
                    rooms_elem = apt.select_one('span.css-1uvpk0b:contains("pokoje")')
                    title_elem = apt.select_one('p.css-14yowm')
                    link_elem = apt.select_one('a.css-n9dffw')
                    
                    # Tutaj trzeba wyciągnąć dane i przetworzyć
                    # Przykładowe przetwarzanie:
                    price = re.sub(r'[^\d]', '', price_elem.text) if price_elem else None
                    area = re.sub(r'[^\d,.]', '', area_elem.text).replace(',', '.') if area_elem else None
                    rooms = re.sub(r'[^\d]', '', rooms_elem.text) if rooms_elem else None
                    title = title_elem.text if title_elem else ""
                    link = link_elem['href'] if link_elem else ""
                    
                    # Odwiedź stronę ogłoszenia, by pobrać więcej szczegółów
                    if link:
                        apartment_details = scrape_apartment_details(link, headers)
                        
                        apartment = {
                            'city': city,
                            'price': int(price) if price else None,
                            'area': float(area) if area else None,
                            'rooms': int(rooms) if rooms else None,
                            'title': title,
                            'link': link,
                            **apartment_details
                        }
                        
                        all_apartments.append(apartment)
                
                except Exception as e:
                    print(f"Error parsing apartment: {e}")
            
            # Dodaj opóźnienie, by nie przeciążać serwera
            time.sleep(random.uniform(1, 3))
            
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
    
    return pd.DataFrame(all_apartments)

def scrape_apartment_details(link, headers):
    """
    Funkcja do zbierania szczegółowych informacji o mieszkaniu z pełnej strony ogłoszenia.
    
    Args:
        link (str): Link do pełnej strony ogłoszenia
        headers (dict): Nagłówki HTTP
        
    Returns:
        dict: Słownik ze szczegółami mieszkania
    """
    full_url = f"https://www.otodom.pl{link}" if not link.startswith('http') else link
    
    try:
        response = requests.get(full_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        details = {}
        
        # Wyciągnij dzielnicę
        location_elem = soup.select_one('a.css-1i90gvc')
        if location_elem:
            location_parts = location_elem.text.split(',')
            if len(location_parts) > 1:
                details['district'] = location_parts[1].strip().lower().replace(' ', '_')
        
        # Wyciągnij rok budowy
        build_year_elem = soup.select_one('div.css-1qzsnkz:contains("Rok budowy")')
        if build_year_elem:
            year_text = build_year_elem.find_next('div').text
            year_match = re.search(r'\d{4}', year_text)
            if year_match:
                details['buildYear'] = int(year_match.group())
        
        # Wyciągnij piętro
        floor_elem = soup.select_one('div.css-1qzsnkz:contains("Piętro")')
        if floor_elem:
            floor_text = floor_elem.find_next('div').text.lower()
            if 'parter' in floor_text:
                details['floor'] = 0
            elif 'podziemne' in floor_text:
                details['floor'] = -1
            else:
                floor_match = re.search(r'\d+', floor_text)
                if floor_match:
                    details['floor'] = int(floor_match.group())
        
        # Wyciągnij stan
        condition_elem = soup.select_one('div.css-1qzsnkz:contains("Stan wykończenia")')
        if condition_elem:
            condition_text = condition_elem.find_next('div').text.lower()
            if 'do wykończenia' in condition_text or 'deweloperski' in condition_text:
                details['condition'] = 'nowe'
            elif 'do remontu' in condition_text:
                details['condition'] = 'do_remontu'
            elif 'dobry' in condition_text:
                details['condition'] = 'dobry'
            else:
                details['condition'] = 'bardzo_dobry'
        
        # Wyciągnij typ budynku
        type_elem = soup.select_one('div.css-1qzsnkz:contains("Rodzaj zabudowy")')
        if type_elem:
            type_text = type_elem.find_next('div').text.lower()
            if 'blok' in type_text:
                details['type'] = 'blok'
            elif 'kamienica' in type_text:
                details['type'] = 'kamienica'
            elif 'apartamentowiec' in type_text:
                details['type'] = 'apartamentowiec'
            else:
                details['type'] = 'dom_wielorodzinny'
        
        # Sprawdź dodatkowe udogodnienia
        details['balcony'] = 1 if 'balkon' in soup.text.lower() else 0
        details['parking'] = 1 if 'miejsce parkingowe' in soup.text.lower() or 'garaż' in soup.text.lower() else 0
        details['elevator'] = 1 if 'winda' in soup.text.lower() else 0
        
        # Dodaj opóźnienie, by nie przeciążać serwera
        time.sleep(random.uniform(0.5, 1.5))
        
        return details
        
    except Exception as e:
        print(f"Error scraping apartment details: {e}")
        return {}

def main():
    """
    Główna funkcja zbierająca dane z kilku miast i zapisująca je do pliku CSV
    """
    # Lista miast do przeszukania
    cities = ['warszawa', 'krakow', 'wroclaw', 'poznan', 'gdansk', 'lodz', 'katowice']
    
    # Słownik przechowujący dane dla każdego miasta
    all_data = {}
    
    # Zbierz dane dla każdego miasta
    for city in cities:
        print(f"\nZbieranie danych dla miasta: {city.upper()}")
        df = scrape_otodom(pages=5, city=city)  # Możesz zwiększyć liczbę stron
        
        # Zapisz dane dla miasta
        all_data[city] = df
        
        # Zapisz dane dla miasta
        df.to_csv(f'mieszkania_{city}.csv', index=False, encoding='utf-8')
        print(f"Zapisano {len(df)} ogłoszeń dla miasta {city}")
    
    # Połącz wszystkie dane w jeden DataFrame
    full_dataset = pd.concat(all_data.values(), ignore_index=True)
    
    # Zapisz pełen zbiór danych
    full_dataset.to_csv('mieszkania_pelny_zbior.csv', index=False, encoding='utf-8')
    print(f"\nZapisano pełen zbiór danych: {len(full_dataset)} ogłoszeń")
    
    # Podstawowa analiza danych
    analyze_dataset(full_dataset)

def analyze_dataset(df):
    """
    Funkcja analizująca zebrany zbiór danych i wyświetlająca podstawowe statystyki
    
    Args:
        df (DataFrame): DataFrame z danymi o mieszkaniach
    """
    print("\n===== ANALIZA ZBIORU DANYCH =====")
    
    # Wyświetl podstawowe informacje o zbiorze danych
    print(f"\nLiczba ogłoszeń: {len(df)}")
    print(f"Liczba kolumn: {len(df.columns)}")
    
    # Wyświetl liczebność ogłoszeń według miast
    print("\nLiczba ogłoszeń według miast:")
    city_counts = df['city'].value_counts()
    for city, count in city_counts.items():
        print(f"  - {city.title()}: {count}")
    
    # Sprawdź braki danych
    print("\nBraki danych:")
    missing_data = df.isnull().sum()
    for column, missing in missing_data.items():
        if missing > 0:
            percentage = (missing / len(df)) * 100
            print(f"  - {column}: {missing} ({percentage:.2f}%)")
    
    # Statystyki cenowe
    print("\nStatystyki cenowe:")
    price_stats = df['price'].describe()
    print(f"  - Średnia cena: {price_stats['mean']:.2f} PLN")
    print(f"  - Mediana ceny: {price_stats['50%']:.2f} PLN")
    print(f"  - Min. cena: {price_stats['min']:.2f} PLN")
    print(f"  - Max. cena: {price_stats['max']:.2f} PLN")
    
    # Statystyki powierzchni
    print("\nStatystyki powierzchni:")
    area_stats = df['area'].describe()
    print(f"  - Średnia powierzchnia: {area_stats['mean']:.2f} m²")
    print(f"  - Mediana powierzchni: {area_stats['50%']:.2f} m²")
    print(f"  - Min. powierzchnia: {area_stats['min']:.2f} m²")
    print(f"  - Max. powierzchnia: {area_stats['max']:.2f} m²")
    
    # Cena za metr kwadratowy
    df['price_per_sqm'] = df['price'] / df['area']
    
    print("\nStatystyki ceny za m²:")
    price_per_sqm_stats = df['price_per_sqm'].describe()
    print(f"  - Średnia cena za m²: {price_per_sqm_stats['mean']:.2f} PLN/m²")
    print(f"  - Mediana ceny za m²: {price_per_sqm_stats['50%']:.2f} PLN/m²")
    
    # Cena za m² według miast
    print("\nŚrednia cena za m² według miast:")
    avg_price_by_city = df.groupby('city')['price_per_sqm'].mean().sort_values(ascending=False)
    for city, avg_price in avg_price_by_city.items():
        print(f"  - {city.title()}: {avg_price:.2f} PLN/m²")



def prepare_for_modeling(df):
    """
    Przygotowuje dane do modelowania
    
    Args:
        df (DataFrame): Oczyszczony DataFrame
        
    Returns:
        tuple: (X, y) gdzie X to cechy, a y to etykiety (ceny)
    """
    # Zdefiniuj cechy do wykorzystania w modelu
    features = [
        'city', 'district', 'area', 'rooms', 'floor', 
        'building_age', 'condition', 'type', 
        'balcony', 'parking', 'elevator',
        'log_area', 'room_density', 'amenities_index'
    ]
    
    # Wybierz tylko cechy, które istnieją w DataFrame
    available_features = [f for f in features if f in df.columns]
    
    # Cechy (X) i etykiety (y)
    X = df[available_features]
    y = df['price']
    
    return X, y

if __name__ == "__main__":
    main()




    # data_collector.py - Skrypt do zbierania danych o mieszkaniach z Otodom
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import logging

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_soup(url):
    """Pobiera zawartość strony przy użyciu Selenium dla JavaScript"""
    #--- DOPISAC ścieżkę do ChromeDriver ---
    chrome_driver_path = "ścieżka/do/chromedriver"
    
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-blink-features=AutomationControlled")
    
    #--- DOPISAC listę User-Agentów ---
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36...",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15..."
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    driver = webdriver.Chrome(executable_path=chrome_driver_path, options=options)
    driver.get(url)
    time.sleep(random.uniform(2, 4))
    html = driver.page_source
    driver.quit()
    return BeautifulSoup(html, 'html.parser')

def scrape_otodom(pages=5, city="warszawa"):
    base_url = "https://www.otodom.pl/pl/oferty/sprzedaz/mieszkanie/"
    
    #--- DOPISAC mapowanie miast na ID regionu ---
    city_ids = {
        "warszawa": "city_id",
        "krakow": "city_id",
        # ...
    }
    
    #--- DOPISAC parametry API ---
    API_URL = "https://www.otodom.pl/api/v2/offers/listing"
    params = {
        "limit": 24,
        "category": "flat",
        "regionId": city_ids.get(city, ""),
        "page": 1
    }
    
    headers = {
        #--- DOPISAC aktualne nagłówki ---
        "authority": "www.otodom.pl",
        "accept": "application/json",
        "referer": f"{base_url}{city}",
    }
    
    #--- DOPISAC dane proxy ---
    proxies = {
        "http": "http://user:pass@ip:port",
        "https": "http://user:pass@ip:port"
    }
    
    all_apartments = []
    
    for page in range(1, pages + 1):
        try:
            params["page"] = page
            response = requests.get(
                API_URL,
                headers=headers,
                params=params,
                proxies=proxies,
                timeout=10
            )
            
            if response.status_code != 200:
                logger.error(f"Błąd API: {response.status_code}")
                continue
                
            data = response.json()
            
            #--- DOPISAC ścieżkę do danych w JSON ---
            for offer in data.get('embedded', {}).get('offers', []):
                try:
                    apartment = {
                        'city': city,
                        'price': offer.get('price'),
                        'area': offer.get('area'),
                        'rooms': offer.get('rooms_num'),
                        'title': offer.get('title'),
                        'link': offer.get('_links', {}).get('self', {}).get('href'),
                        #--- DOPISAC dodatkowe pola z API ---
                    }
                    all_apartments.append(apartment)
                
                except Exception as e:
                    logger.error(f"Błąd przetwarzania oferty: {e}")
            
            time.sleep(random.gammavariate(alpha=9, beta=0.4))
            
        except Exception as e:
            logger.error(f"Błąd strony {page}: {e}")
    
    return pd.DataFrame(all_apartments)

def scrape_apartment_details(link):
    try:
        soup = get_soup(link)
        
        details = {}
        
        #--- DOPISAC aktualne selektory dla szczegółów ---
        location_section = soup.find('div', {'aria-label': 'Lokalizacja'})
        if location_section:
            details['district'] = location_section.find('a').text.strip()
        
        # Przykładowy selektor dla roku budowy
        year_element = soup.find('div', text=re.compile(r'Rok budowy', re.I))
        if year_element:
            details['buildYear'] = year_element.find_next('div').text.strip()
        
        #--- DOPISAC pozostałe selektory ---
        # floor_element = ...
        # condition_element = ...
        # type_element = ...
        
        return details
        
    except Exception as e:
        logger.error(f"Błąd szczegółów: {e}")
        return {}

def main():
    cities = ['warszawa', 'krakow', 'wroclaw', 'poznan', 'gdansk']
    
    all_data = pd.DataFrame()
    
    for city in cities:
        logger.info(f"Zbieranie danych dla: {city}")
        df = scrape_otodom(pages=3, city=city)
        
        # Dodawanie szczegółów
        df['details'] = df['link'].apply(lambda x: scrape_apartment_details(x))
        df = pd.concat([df, df['details'].apply(pd.Series)], axis=1)
        
        all_data = pd.concat([all_data, df], ignore_index=True)
        time.sleep(10)
    
    # Czyszczenie danych
    clean_df = clean_dataset(all_data)
    
    #--- DOPISAC nazwę pliku wyjściowego ---
    clean_df.to_csv('otodom_data.csv', index=False)
    logger.info(f"Zapisano {len(clean_df)} ofert")

def clean_dataset(df):
    """
    Funkcja czyszcząca i przygotowująca dane do modelowania
    
    Args:
        df (DataFrame): DataFrame z surowymi danymi
        
    Returns:
        DataFrame: Oczyszczony DataFrame gotowy do modelowania
    """
    # Zrób kopię danych
    clean_df = df.copy()
    
    # Usuń wiersze z brakującymi wartościami w kluczowych kolumnach
    key_columns = ['price', 'area', 'rooms', 'city']
    clean_df = clean_df.dropna(subset=key_columns)
    
    # Usuń duplikaty
    clean_df = clean_df.drop_duplicates()
    
    # Usuń odstające wartości cenowe (metoda IQR)
    Q1 = clean_df['price'].quantile(0.25)
    Q3 = clean_df['price'].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    clean_df = clean_df[(clean_df['price'] >= lower_bound) & (clean_df['price'] <= upper_bound)]
    
    # Usuń odstające wartości powierzchni
    Q1_area = clean_df['area'].quantile(0.25)
    Q3_area = clean_df['area'].quantile(0.75)
    IQR_area = Q3_area - Q1_area
    
    lower_bound_area = Q1_area - 1.5 * IQR_area
    upper_bound_area = Q3_area + 1.5 * IQR_area
    
    clean_df = clean_df[(clean_df['area'] >= lower_bound_area) & (clean_df['area'] <= upper_bound_area)]
    
    # Przekształć wartości kategoryczne
    # Wypełnij brakujące wartości w kolumnach kategorycznych
    categorical_columns = ['district', 'condition', 'type']
    for col in categorical_columns:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].fillna('unknown')
    
    # Wypełnij brakujące wartości w kolumnach numerycznych
    numerical_columns = ['buildYear', 'floor', 'balcony', 'parking', 'elevator']
    for col in numerical_columns:
        if col in clean_df.columns:
            if col == 'buildYear':
                clean_df[col] = clean_df[col].fillna(clean_df[col].median())
            elif col == 'floor':
                clean_df[col] = clean_df[col].fillna(0)  # Załóżmy, że brakujące piętra to parter
            else:
                clean_df[col] = clean_df[col].fillna(0)  # Zakładamy brak udogodnienia
    
    # Utwórz nowe cechy
    # Wiek budynku
    if 'buildYear' in clean_df.columns:
        current_year = 2025  # Aktualizuj w razie potrzeby
        clean_df['building_age'] = current_year - clean_df['buildYear']
    
    # Logarytm powierzchni (często ma liniową zależność z logarytmem ceny)
    clean_df['log_area'] = np.log1p(clean_df['area'])
    
    # Cena za m²
    clean_df['price_per_sqm'] = clean_df['price'] / clean_df['area']
    
    # Wskaźnik gęstości pokoi (liczba pokoi na m²)
    clean_df['room_density'] = clean_df['rooms'] / clean_df['area']
    
    # Wskaźnik udogodnień (suma balkon, parking, winda)
    amenities_cols = ['balcony', 'parking', 'elevator']
    if all(col in clean_df.columns for col in amenities_cols):
        clean_df['amenities_index'] = clean_df[amenities_cols].sum(axis=1)
    
    # Usuń oferty bez kluczowych parametrów
    df = df.dropna(subset=['price', 'area', 'rooms'])
    
    # Filtruj ekstremalne wartości
    df = df[(df['price'] > 10000) & (df['price'] < 5000000)]
    df = df[(df['area'] > 15) & (df['area'] < 500)]
    
    return df

if __name__ == "__main__":
    main()