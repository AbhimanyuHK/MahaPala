"""

Created by abhimanyu at 15/12/23

"""

import streamlit as st
from PIL import Image

from mahapala import FruitsClassification

st.set_page_config(page_title="MahaPala", layout="wide")


@st.cache_resource
def get_classifier():
    classifier = FruitsClassification()
    classifier.fit()
    return classifier


st.title("MahaPala")

upload_col, result_col = st.columns(2)

with upload_col:
    st.info("Upload a fruit / plant to identify")
    uploaded_file = st.file_uploader("Drag and Drop or Select Files", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name)

with result_col:
    st.info("Predication Result")
    if uploaded_file is not None:
        with st.spinner("Predicting..."):
            img = Image.open(uploaded_file).resize((224, 224))
            result = get_classifier().predict_image(img)

        st.subheader(uploaded_file.name)
        for x in result:
            confidence = float(x["confidence"].split(" ")[0])
            st.write(f"Class : {x['class']}")
            st.progress(min(int(confidence), 100), text=f"{confidence}%")
