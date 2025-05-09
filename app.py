import streamlit as st

# Initialize connection.
conn = st.connection('mysql', type='sql')

# Perform query.
df = conn.query('SELECT * from cars;', ttl=600)

# Display the data
st.dataframe(df)
