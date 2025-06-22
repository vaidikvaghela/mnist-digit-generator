import streamlit as st
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

# App title
st.title("Handwritten Digit Generator")

# User input
digit = st.number_input("Enter a digit (0–9):", min_value=0, max_value=9, value=0, step=1)

if st.button("Generate Images"):
    st.write(f"5 handwritten samples of digit {digit}:")
    
    # Filter for selected digit
    indices = np.where(y_train == digit)[0]
    samples = np.random.choice(indices, 5, replace=False)
    
    # Plot images
    fig, axes = plt.subplots(1, 5, figsize=(10, 2))
    for i, idx in enumerate(samples):
        axes[i].imshow(x_train[idx], cmap='gray')
        axes[i].axis('off')
    
    st.pyplot(fig)
