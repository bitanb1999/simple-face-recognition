import os
import shutil
import streamlit as st


st.header("Settings🔨")

# Function to reset data
def reset_data():
    with st.spinner('Removing data...'):
        list_dir = os.listdir('static')
        if not list_dir:
            st.info("Data doesn't exists.")
        else:
            for dir in list_dir:
                dir_remove = os.path.join('static', dir)
                if os.path.exists(dir_remove):
                    shutil.rmtree(dir_remove)
                    os.makedirs(dir_remove)
            else:
                st.success('Data removed successfully!')

# Check if confirmation flag is set in session state
if 'reset_confirmed' not in st.session_state:
    st.session_state.reset_confirmed = False

# Button to trigger reset confirmation
if st.button("Reset Data"):
    st.session_state.reset_confirmed = True

# If user has triggered reset confirmation
if st.session_state.reset_confirmed:
    container = st.container(border=True)
    container.write('Are you sure?')
    if container.button('Yes', type='primary', key='confirm_yes', use_container_width=True):
        reset_data()
        st.session_state.reset_confirmed = False
    if container.button('No', type='secondary', key='confirm_no', use_container_width=True):
        st.write('Okay.')
        st.session_state.reset_confirmed = False
