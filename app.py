import streamlit as st
import pandas as pd
import os
import sqlite3
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def load_data_from_db():
    conn = sqlite3.connect('ads.db')
    query = """
        SELECT category AS Category, date_scrape AS Date, count AS Count
        FROM categories
    """
    df = pd.read_sql(query, conn, parse_dates=['Date'])
    conn.close()
    return df

def filter_data_by_date_range(df, date_range):
    today = pd.Timestamp.today().normalize()
    
    if date_range == "Last 7 days":
        start_date = today - pd.Timedelta(days=7)
    elif date_range == "Last month":
        start_date = today - pd.Timedelta(days=30)
    elif date_range == "Last 3 months":
        start_date = today - pd.Timedelta(days=90)
    elif date_range == "Last year":
        start_date = today - pd.Timedelta(days=365)
    else:  # "All"
        return df
    
    return df[df['Date'] >= start_date]

def plot_trends(df, selected_categories=None): 
    if selected_categories:
        df = df[df['Category'].isin(selected_categories)]
    
    df_aggregated = df.groupby(['Date', 'Category'], as_index=False)['Count'].sum()
    
    # Convert 'Count' to numeric (in case it's stored as a string)
    df_aggregated['Count'] = pd.to_numeric(df_aggregated['Count'], errors='coerce').fillna(0)
    
    # Pivot the DataFrame
    pivot_df = df_aggregated.pivot(index='Date', columns='Category', values='Count')
    
     # Sort index (dates) in ascending order
    pivot_df = pivot_df.sort_index(ascending=True)
    
    st.line_chart(pivot_df)


def display_table(df):
    if df.empty:
        st.warning("No data available for selected filters.")
        return
        
    latest_date = df['Date'].max()
    latest = df[df['Date'] == latest_date]
    
    previous_dates = sorted(df['Date'].unique())
    if len(previous_dates) > 1:
        previous_date_idx = previous_dates.index(latest_date) - 1
        previous_date = previous_dates[previous_date_idx]
        previous = df[df['Date'] == previous_date]
        
        merged = latest.merge(previous, on='Category', suffixes=('_today', '_yesterday'), how='left')
        merged['Trend'] = merged.apply(lambda row: 
            '⬆️' if row['Count_today'] > row['Count_yesterday'] 
            else ('⬇️' if row['Count_today'] < row['Count_yesterday'] else '➡️'), axis=1)
        
        st.dataframe(merged[['Category', 'Count_today', 'Trend']])
    else:
        st.dataframe(latest[['Category', 'Count']].rename(columns={'Count': 'Count_today'}))
        st.info("Not enough data in the selected range to show trends.")

def fetch_avg_price_per_year():
    conn = sqlite3.connect('ads.db')
    query = """
        SELECT car_year, ROUND(AVG(CAST(price AS REAL)),0) as avg_price, 
        ROUND(AVG(CAST(car_km AS INTEGER)),0) as avg_km,
        COUNT(id) as ads_count
        FROM cars
        GROUP BY car_year
        ORDER BY car_year;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ("Home", "About"))

if page == "Home":
    st.title("Welcome to the Car Ads Dashboard")
    st.title("Dacia Vehicles Trend Analysis")

    df = load_data_from_db()

    if not df.empty:
        df['Date'] = pd.to_datetime(df['Date'])
        
        st.subheader("Data Filters")
        
        date_ranges = ["Last 7 days", "Last month", "Last 3 months", "Last year", "All"]
        selected_date_range = st.selectbox("Select Date Range", options=date_ranges, index=4)
        
        filtered_df = filter_data_by_date_range(df, selected_date_range)
        
        all_categories = df['Category'].unique()
        default_categories = all_categories
        selected_categories = st.multiselect("Select Categories to Display", 
                                            options=all_categories, 
                                            default=default_categories)
        
        st.subheader("Current Trends")
        category_filtered_df = filtered_df[filtered_df['Category'].isin(selected_categories)] if selected_categories else filtered_df
        display_table(category_filtered_df)
        
        st.subheader("Trend Chart")
        plot_trends(filtered_df, selected_categories)

    st.title("Average Asking Price by First Registration Year")
    df = fetch_avg_price_per_year()
    if not df.empty:
        # Create figure
        fig = go.Figure()

        # Add bar for avg_price
        fig.add_trace(go.Bar(
            x=df["car_year"], 
            y=df["avg_price"], 
            name="Average Price (€)",
            marker_color="blue",
        ))

        # Add bar for avg_km
        fig.add_trace(go.Bar(
            x=df["car_year"], 
            y=df["avg_km"], 
            name="Average KM",
            marker_color="green",
        ))

        # Add line for ads_count (with a secondary Y-axis)
        fig.add_trace(go.Scatter(
            x=df["car_year"], 
            y=df["ads_count"], 
            name="Number of Ads", 
            mode="lines+markers",
            yaxis="y2",
            marker_color="red"
        ))

        # Update layout for dual axes
        fig.update_layout(
            title="Average Price, KM & Number of Ads per Year",
            xaxis_title="Year",
            yaxis=dict(
                title="Price (€) & KM",
                side="left",
            ),
            yaxis2=dict(
                title="Number of Ads",
                overlaying="y",
                side="right",
            ),
            barmode="group",
            legend_title="Metrics"
        )

        # Display in Streamlit
        st.plotly_chart(fig)

        # Display in dataframe 
        st.subheader("Data")
        st.dataframe(df)

    else:
        st.write("No data available for displaying the bar chart.")

if page == "About":
    st.title("About Page")
    

    
