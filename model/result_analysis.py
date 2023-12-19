import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from amino_nn import AminoAcidNN
import sys


def visualize_correlations(df, y_only=False): 
    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    ### BAR PLOT:
    if y_only:
        # Extract the last row (correlations with the output)
        output_correlations = correlation_matrix.iloc[-1, :-1]

        # Create a bar plot for the output correlations
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(x=output_correlations.index, y=output_correlations.values, hue=output_correlations.index, palette='coolwarm', legend=False)

        # Add a horizontal line at 0
        ax.axhline(0, color='black', linestyle='--', linewidth=1)

        # Label each bar with its value
        for i, val in enumerate(output_correlations.values):
            ax.text(i, val, f'{val:.2f}', ha='center', va='bottom')
    ###

    ### HEATMAP:
    else:
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    ###
        
    plt.title('Correlation Bar Plot for Amino Acid Counts and Contagiousness Score')
    plt.xlabel('Amino Acids')
    plt.show()


def main():
    """
    Analyzes input data and generates visualizations.
    Usage: python3 result_analysis.py <filename>
    Example:
    python3 result_analysis.py ../data/processed_input_data_vacmodel.csv
    """
    # Get input filename from command-line arguments, defaulting to a provided path
    modelname = sys.argv[1] if len(sys.argv) > 1 else './model_storage/contagiousness_model0.pt'
    
    # Load the entire model
    model = torch.load(modelname)

    # Instantiate a new instance of the model class
    input_size = 21
    hidden_sizes = [256, 128, 64]
    output_size = 1
    new_model = AminoAcidNN(input_size, hidden_sizes, output_size)

    # Load the state dictionary from the loaded model
    new_model.load_state_dict(model)

    # Set the new model to evaluation mode
    new_model.eval()

    # Generate input data
    num_inputs = 21
    input_data = torch.zeros(num_inputs, num_inputs)
    for i in range(num_inputs):
        input_data[i, i] = 100  # Set one input to 100

    # Make predictions using the new model
    with torch.no_grad():
        output_data = new_model(input_data)

    # Convert PyTorch tensors to NumPy arrays
    input_data_np = input_data.numpy()
    output_data_np = output_data.numpy()

    # Create a DataFrame for visualization
    columns = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'Other'] + ['Contagiousness_Score']
    df = pd.DataFrame(data=np.hstack((input_data_np, output_data_np)), columns=columns)

    # Visualize correlations as a heatmap (y_only=False) or barplot (y_only=True)
    visualize_correlations(df, y_only=True)


if __name__ == "__main__":
    main()
