import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST dataset (safe on Streamlit Cloud)
@st.cache_data
def load_data():
    data = fetch_openml('mnist_784', version=1, as_frame=False)
    images = data.data.reshape(-1, 28, 28)
    labels = data.target.astype(np.int8)
    return images, labels

X, y = load_data()

st.title("🧠 Handwritten Digit Generator (0–9)")

digit = st.number_input("Select a digit", 0, 9, 0)

if st.button("Generate Samples"):
    indices = np.where(y == digit)[0]
    selected = np.random.choice(indices, 5, replace=False)

    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, idx in enumerate(selected):
        axes[i].imshow(X[idx], cmap="gray")
        axes[i].axis("off")
    st.pyplot(fig)
