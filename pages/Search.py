import streamlit as st
import sqlite3
import pandas as pd
import numpy as np

# Connect to your SQLite database (replace with your actual database path)
conn = sqlite3.connect('ads.db')
cursor = conn.cursor()

# Function to load data from the SQLite database into a pandas DataFrame
def load_data():
    query = "SELECT * FROM cars"
    df = pd.read_sql(query, conn)
    return df

# Load data
df = load_data()

#Column Transformation
# Convert car_year column to integers, discarding non-integer values
df['car_year'] = pd.to_numeric(df['car_year'], errors='coerce').dropna().astype(int)



# Streamlit App UI
st.title("Car Ads Filter")

# Filters
st.sidebar.header("Filters")

# Filter: ID
id_filter = st.sidebar.text_input("ID", "")

# Filter: Ad Type (Distinct Values)
ad_type_options = df['ad_type'].unique().tolist()
ad_type_filter = st.sidebar.selectbox("Ad Type", ['All'] + ad_type_options)

# Filter: Car KM (Slider)
car_km_min, car_km_max = df['car_km'].apply(pd.to_numeric, errors='coerce').dropna().min(), df['car_km'].apply(pd.to_numeric, errors='coerce').dropna().max()
car_km_filter = st.sidebar.slider("Car KM", min_value=int(car_km_min), max_value=int(car_km_max), value=(int(car_km_min), int(car_km_max)))

# Filter: Car Year (Slider)
car_year_min, car_year_max = df['car_year'].astype(int).min(), df['car_year'].astype(int).max()
car_year_filter = st.sidebar.slider("Car Year", min_value=int(car_year_min), max_value=int(car_year_max), value=(int(car_year_min), int(car_year_max)))

# Filter: Date Posted (Slider)
df['date_posted'] = pd.to_datetime(df['date_posted'])
date_posted_min, date_posted_max = df['date_posted'].min(), df['date_posted'].max()
date_posted_filter = st.sidebar.slider("Date Posted", min_value=date_posted_min, max_value=date_posted_max, value=(date_posted_min, date_posted_max))

# Filter: Title (Text Box)
title_filter = st.sidebar.text_input("Title", "")

# Filter: Price (Slider)
price_min, price_max = df['price'].apply(pd.to_numeric, errors='coerce').dropna().min(), df['price'].apply(pd.to_numeric, errors='coerce').dropna().max()
price_filter = st.sidebar.slider("Price", min_value=int(price_min), max_value=int(price_max), value=(int(price_min), int(price_max)))

# Apply Filters
filtered_df = df

# Apply ID filter
if id_filter:
    filtered_df = filtered_df[filtered_df['id'].str.contains(id_filter, case=False)]

# Apply Ad Type filter
if ad_type_filter != 'All':
    filtered_df = filtered_df[filtered_df['ad_type'] == ad_type_filter]

# Apply Car KM filter
filtered_df = filtered_df[(filtered_df['car_km'].apply(pd.to_numeric, errors='coerce') >= car_km_filter[0]) & 
                           (filtered_df['car_km'].apply(pd.to_numeric, errors='coerce') <= car_km_filter[1])]

# Apply Car Year filter
filtered_df = filtered_df[(filtered_df['car_year'].astype(int) >= car_year_filter[0]) & 
                           (filtered_df['car_year'].astype(int) <= car_year_filter[1])]

# Apply Date Posted filter
filtered_df = filtered_df[(filtered_df['date_posted'] >= pd.to_datetime(date_posted_filter[0])) & 
                           (filtered_df['date_posted'] <= pd.to_datetime(date_posted_filter[1]))]

# Apply Title filter
if title_filter:
    filtered_df = filtered_df[filtered_df['title'].str.contains(title_filter, case=False)]

# Apply Price filter
filtered_df = filtered_df[(filtered_df['price'].apply(pd.to_numeric, errors='coerce') >= price_filter[0]) & 
                           (filtered_df['price'].apply(pd.to_numeric, errors='coerce') <= price_filter[1])]

# Display filtered dataframe
st.subheader("Filtered Car Ads")
st.dataframe(filtered_df)

# Option to download filtered data as CSV
st.download_button("Download CSV", filtered_df.to_csv(index=False), "filtered_car_ads.csv", "text/csv")
