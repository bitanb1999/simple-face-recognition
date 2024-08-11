import streamlit as st


st.title("🤳 Simple Face Recognition")

welcome = st.Page(
    "welcome.py",
    title="Welcome",
    icon=":material/home:",
)
verify = st.Page(
    "verify.py",
    title="Verify Attendance",
    icon=":material/security:",
)
register = st.Page(
    "register.py", 
    title="Register Employee", 
    icon=":material/person_add:",
)

pg = st.navigation({'Menu': [welcome, verify, register]})

pg.run()

