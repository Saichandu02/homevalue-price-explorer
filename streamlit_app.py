"""Optional Streamlit front-end (NOT the verified UI).

The verified UI is the Flask page at `/`. This optional app calls the same
API; it requires `pip install -r requirements-optional.txt` and a running
API (default http://127.0.0.1:8101). It was not executed on the build VM
(large Streamlit install avoided there) — see docs/decisions.md.
"""
import os
import requests
import streamlit as st

API = os.environ.get("HOMEVALUE_API", "http://127.0.0.1:8101")
st.title("HomeValue — Housing Price Explorer")
med_inc = st.number_input("Median income (10k USD)", value=5.0)
payload = {
    "MedInc": med_inc,
    "HouseAge": st.number_input("House age", value=20.0),
    "AveRooms": st.number_input("Avg rooms", value=6.0),
    "AveBedrms": st.number_input("Avg bedrooms", value=1.0),
    "Population": st.number_input("Population", value=1000.0),
    "AveOccup": st.number_input("Avg occupants", value=3.0),
    "Latitude": st.number_input("Latitude", value=37.88),
    "Longitude": st.number_input("Longitude", value=-122.23),
}
if st.button("Predict"):
    r = requests.post(f"{API}/api/predict", json=payload, timeout=10)
    st.json(r.json())
