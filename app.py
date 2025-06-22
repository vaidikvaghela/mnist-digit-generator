import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

# Load MNIST using sklearn (safe for Streamlit Cloud)
@st.cache_data
def load_mnist():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'].reshape(-1, 28, 28)
    y = mnist['target'].astype(int)
    return X, y

X, y = load_mnist()

st.title("Handwritten Digit Generator (0–9)")

digit = st.number_input("Enter a digit (0–9):", min_value=0, max_value=9, value=0, step=1)

if st.button("Generate Images"):
    st.write(f"5 handwritten samples of digit {digit}:")
    
    # Find indices for the digit
    indices = np.where(y == digit)[0]
    chosen = np.random.choice(indices, 5, replace=False)

    # Plot images
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, idx in enumerate(chosen):
        axes[i].imshow(X[idx], cmap='gray')
        axes[i].axis('off')
    
    st.pyplot(fig)
