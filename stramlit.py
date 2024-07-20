import streamlit as st

image = st.camera_input('写真を撮影します')
print(type(image))