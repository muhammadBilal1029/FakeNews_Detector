import streamlit as st
import pickle
import eli5
from eli5.sklearn import explain_prediction_sklearn

# Load model and vectorizer
model_path = 'D:/ML_Projects/FakeNews_Detector/model.pkl'
vec_path = 'D:/ML_Projects/FakeNews_Detector/vectorizer.pkl'

with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(vec_path, 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Fake News Detector with Explanation")

news_text = st.text_area("Enter news content here:")

if st.button("Predict"):
    if not news_text.strip():
        st.warning("Please enter some news content.")
    else:
        vec = vectorizer.transform([news_text])
        pred = model.predict(vec)[0]
        label = "REAL" if pred == 1 else "FAKE"
        st.markdown(f"### Prediction: {label}")

        # Generate explanation
        explanation = eli5.format_as_text(
    eli5.explain_prediction_sklearn(model, vec)
    )
        st.markdown("### Explanation:")
        st.write(explanation, unsafe_allow_html=True)
