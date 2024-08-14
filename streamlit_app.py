import streamlit as st


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

pg = st.navigation({'Menu': [welcome, verify, register, stored]})

pg.run()

