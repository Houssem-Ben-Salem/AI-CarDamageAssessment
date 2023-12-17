
import pandas as pd

def clean_cost(value):
    ''' Function to clean and convert cost values to integers. '''
    value = value.replace(',', '').replace(' euros', '')
    try:
        return int(value)
    except ValueError:
        return None  # Return None for values that cannot be converted to int

def main():
    # Load the CSV file
    file_path = 'global_repair_cost.csv'  # Update with the correct file path
    data = pd.read_csv(file_path)

    # Apply the function to the 'Result' column
    data['Cleaned_Cost'] = data['Result'].apply(clean_cost)

    # Drop rows with None values in 'Cleaned_Cost'
    data.dropna(subset=['Cleaned_Cost'], inplace=True)

    # Defining the bins and labels
    bins = [0, 500, 1000, 1500, float('inf')]  # Upper limit is infinity to include all higher values
    labels = ['Less than 500 euros', 'Between 500 and 1000', 'Between 1000 and 1500', 'More than 1500']
    class_ids = {label: idx for idx, label in enumerate(labels)}  # Mapping labels to class IDs

    # Binning the data
    data['Cost_Range'] = pd.cut(data['Cleaned_Cost'], bins=bins, labels=labels, right=False)

    # Adding a column for class ID
    data['Class_ID'] = data['Cost_Range'].map(class_ids)

    # Preparing the data for CSV export
    data_to_export = data[['Image', 'Cost_Range', 'Class_ID']]

    # Exporting to CSV
    export_file_path = 'repair_cost_ranges.csv'  # Update with the desired file path
    data_to_export.to_csv(export_file_path, index=False)

if __name__ == "__main__":
    main()
