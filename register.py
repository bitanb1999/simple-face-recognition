import os
import streamlit as st

from uuid import uuid4


st.header("Register Employee🙋")

file_name, file_type = 'video1', 'mp4'
file_path = os.path.join('assets', 'placeholder', f"{file_name}.{file_type}")

file_uploaded = st.file_uploader("Drop a AVI/MP4/MOV/MKV file!", accept_multiple_files=False, type=['avi', 'mp4', 'mov', 'mkv'])
if file_uploaded is not None:
    file_name, file_type = file_uploaded.name, file_uploaded.type.split('/')[-1]
    
    file_name_new = f"{str(uuid4().hex)}.{file_type}"
    file_path = os.path.join('static', 'registration', file_name_new)

    with open(file_path, "wb") as f:
        f.write(file_uploaded.getbuffer())
    
    st.success("File saved successfully.")
else:
    st.write("You are using a placeholder video. Upload your video to Register.")

video = open(file_path, "rb")
video = video.read()
st.video(video, autoplay=True, muted=True)
