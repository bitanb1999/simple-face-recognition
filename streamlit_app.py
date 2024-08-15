import os
import streamlit as st

os.makedirs('static/attendance', exist_ok=True)
os.makedirs('static/dataset', exist_ok=True)
os.makedirs('static/embedding', exist_ok=True)
os.makedirs('static/model', exist_ok=True)
os.makedirs('static/registration', exist_ok=True)


st.logo('assets/placeholder/logo.jpg')
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

if not hasattr(st, 'already_started_server'):
    # Hack the fact that Python modules (like st) only load once to
    # keep track of whether this file already ran.
    st.already_started_server = True

    st.write('''
        The first time this script executes it will run forever because it's
        running a Flask server.

        Just close this browser tab and open a new one to see your Streamlit
        app.
    ''')

    from app import create_app

    app = create_app()
    app.run()
