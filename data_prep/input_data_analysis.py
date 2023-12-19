import pandas as pd
import sys


"""
Prints info about input data
Usage: python3 input_data_analysis.py <filename>
i.e.:
python3 input_data_analysis.py ../data/processed_input_data_vacmodel.csv
"""
def main():
    # Get input filename from command-line arguments, defaulting to a provided path
    filename = sys.argv[1] if len(sys.argv) > 1 else '../data/processed_input_data_vacmodel.csv'
    
    # Read the DataFrame from the CSV file
    df = pd.read_csv(filename)

    # Display DataFrame information and save to a text file
    print("DataFrame Info:")
    print(df.info())

    # Calculate and display the standard deviation of 'Contagiousness_Score'
    std_dev_contagiousness = df['Contagiousness_Score'].std()
    std_dev_contagiousness_str = f"\nStandard Deviation of Contagiousness_Score: {std_dev_contagiousness:.2f}"
    print(std_dev_contagiousness_str)

    print(df.describe().round(2))
    # Save descriptive statistics, std deviation, and df_info to a text file
    descriptive_stats = df.describe().round(2)
    output_filename = f'../data/{get_base_filename(filename)}_info.txt'
    with open(output_filename, 'w') as file:
        file.write(f"{std_dev_contagiousness_str}\n\n")
        descriptive_stats.to_csv(file, sep='\t', mode='a', header=True, index=False)

    print(f"Descriptive Statistics saved to {output_filename}")

def get_base_filename(filename):
    # Exclude only the last extension
    return '.'.join(filename.split('.')[:-1])

if __name__ == "__main__":
    main()