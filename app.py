import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from minitorch.datasets import Graph, simple, spiral, xor

def visualize_dataset(dataset_name, X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')
    
    # Manually create classifiers (example: two lines)
    plt.plot([0, 1], [1, 0], 'r--')  # Red line classifier
    plt.plot([0, 1], [0, 1], 'b--')  # Blue line classifier
    
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{dataset_name} Dataset')
    plt.grid(True)
    plt.savefig("dataset_visualization.png")  # Save the image
    st.image("dataset_visualization.png")  # Display the image in Streamlit

def main():
    st.title('Dataset Visualization')
    
    # Dataset selection
    dataset_names = ['Simple', 'Spiral', 'XOR']  # Add more if needed
    selected_dataset = st.selectbox('Select Dataset', dataset_names)
    
    # Visualize the selected dataset
    if selected_dataset == 'Simple':
        N = 100  # Choose the number of points
        dataset = simple(N)
    elif selected_dataset == 'Spiral':
        dataset = spiral(N=1000)  # Choose appropriate N value
    elif selected_dataset == 'XOR':
        dataset = xor(N=1000)  # Choose appropriate N value

    visualize_dataset(selected_dataset, np.array(dataset.X), np.array(dataset.y))

if __name__ == '__main__':
    main()
