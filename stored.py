import os
import pandas as pd
import streamlit as st

from loader import get_encoder


st.header("Stored Data📦")

with st.spinner("Preparing all AI to be ready..."):
    encoder_y_facenet = get_encoder()

names = encoder_y_facenet.classes_.tolist()
ids = encoder_y_facenet.transform(names)

df = pd.DataFrame({'ID': ids, 'Name': names})
df['Train'] = df.Name.apply(
    lambda x: 
        len(os.listdir(f'assets/dataset/train/{x}')) 
        if x in os.listdir('assets/dataset/train') else len(os.listdir(f'static/dataset/train/{x}'))
    )
df['Validation'] = df.Name.apply(
    lambda x: 
        len(os.listdir(f'assets/dataset/val/{x}')) 
        if x in os.listdir('assets/dataset/val') else len(os.listdir(f'static/dataset/val/{x}'))
    )

df.Name = df.Name.apply(lambda x: f"{x[:4]}".ljust(20, '*') if len(x) > 5 else f"{x}".ljust(20, '*'))

st.dataframe(df, hide_index=True)
