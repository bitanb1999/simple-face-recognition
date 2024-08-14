import streamlit as st
import pandas as pd

from loader import get_yolov8n_face, get_facenet_512, get_encoder_y_facenet, get_classifier_facenet


st.header("Stored Data📦")

with st.spinner("Preparing all AI to be ready..."):
    encoder_y_facenet = get_encoder_y_facenet()

names = encoder_y_facenet.classes_.tolist()
ids = encoder_y_facenet.transform(names)

df = pd.DataFrame({'ID': ids, 'Name': names})

st.dataframe(df, hide_index=True)
