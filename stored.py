import streamlit as st
import pandas as pd

from loader import get_encoder


st.header("Stored Data📦")

with st.spinner("Preparing all AI to be ready..."):
    encoder_y_facenet = get_encoder()

names = encoder_y_facenet.classes_.tolist()
ids = encoder_y_facenet.transform(names)

df = pd.DataFrame({'ID': ids, 'Name': names})

st.dataframe(df, hide_index=True)
