"""
Simple test version of the AutoPlot-Digitizer UI
"""

import streamlit as st

st.set_page_config(
    page_title="AutoPlot-Digitizer Test",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AutoPlot-Digitizer Test")
st.write("This is a test version to verify Streamlit is working properly.")

# Test file upload
uploaded_file = st.file_uploader("Test file upload", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"File size: {len(uploaded_file.read())} bytes")

    # Reset file pointer
    uploaded_file.seek(0)

    # Display image if it's an image file
    if uploaded_file.type.startswith('image'):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

st.write("✅ Streamlit UI is working correctly!")
