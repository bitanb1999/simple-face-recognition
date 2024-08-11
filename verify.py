import os
import streamlit as st

from uuid import uuid4


st.header("Verify Attendance🔐")

file_name, file_type = 'img1', 'jpg'
file_path = os.path.join('assets', 'placeholder', f"{file_name}.{file_type}")

file_uploaded = st.file_uploader("Drop a JPG/PNG file!", accept_multiple_files=False, type=['jpg', 'png'])
if file_uploaded is not None:
    file_name, file_type = file_uploaded.name.split('.')
    
    file_name_new = f"{str(uuid4().hex)}.{file_type}"
    file_path = os.path.join('static', 'attendance', file_name_new)

    with open(file_path, "wb") as f:
        f.write(file_uploaded.getbuffer())
    
    st.success("File saved successfully.")
else:
    st.write("You are using a placeholder image. Upload your image to Verify Attendance.")

st.image(file_path, caption=f"{file_name}.{file_type}", use_column_width='always', channels='BGR')
