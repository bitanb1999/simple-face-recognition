import os
import streamlit as st

os.makedirs('static/attendance', exist_ok=True)
os.makedirs('static/dataset', exist_ok=True)
os.makedirs('static/embedding', exist_ok=True)
os.makedirs('static/model', exist_ok=True)
os.makedirs('static/registration', exist_ok=True)


st.title("🤳 Simple Face Recognition")

welcome = st.Page(
    "welcome.py",
    title="Welcome",
    icon=":material/home:",
)
verify = st.Page(
    "verify.py",
    title="Verify Face",
    icon=":material/security:",
)
register = st.Page(
    "register.py", 
    title="Register Face", 
    icon=":material/person_add:",
)
stored = st.Page(
    'stored.py',
    title='Stored Data',
    icon=':material/database:',
)
settings = st.Page(
    'settings.py',
    title='Settings',
    icon=':material/settings:'
)

pg = st.navigation({'Menu': [welcome, verify, register, stored, settings]})

pg.run()

