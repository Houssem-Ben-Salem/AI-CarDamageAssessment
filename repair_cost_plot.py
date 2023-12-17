
import pandas as pd
import matplotlib.pyplot as plt

def clean_cost(value):
    ''' Function to clean and convert cost values to integers. '''
    value = value.replace(',', '').replace(' euros', '')
    try:
        return int(value)
    except ValueError:
        return None  # Return None for values that cannot be converted to int

def main():
    # Load the CSV file
    file_path = 'repair_cost_2.csv'  # Update with the correct file path
    data = pd.read_csv(file_path)

    # Apply the function to the 'Result' column
    data['Cleaned_Cost'] = data['Result'].apply(clean_cost)

    # Drop rows with None values in 'Cleaned_Cost'
    data.dropna(subset=['Cleaned_Cost'], inplace=True)

    # Counting the frequency of each unique cost value
    cleaned_cost_counts = data['Cleaned_Cost'].value_counts().sort_index()
    cleaned_cost_counts.index = cleaned_cost_counts.index.astype(int)

    # Plotting the frequency of each cost value
    plt.figure(figsize=(12, 6))
    cleaned_cost_counts.plot(kind='bar')
    plt.title('Frequency of Each Repair Cost')
    plt.xlabel('Repair Cost (Euros)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
