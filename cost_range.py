import pandas as pd
import plotly.express as px

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

    # Define new bin edges and labels after merging the first two bins
    bins = [0, 1000, 1500, 10000, float('inf')]
    labels = [
        'Less than 1000', 
        '1000-1500', 
        '1500-10000', 
        'More than 10000'
    ]

    # Binning the data with the new bins
    data['Cost_Range'] = pd.cut(data['Cleaned_Cost'], bins=bins, labels=labels, right=False)
    class_ids = {label: idx for idx, label in enumerate(labels)}  # Mapping labels to class IDs

    # Adding a column for class ID
    data['Class_ID'] = data['Cost_Range'].map(class_ids)

    # Preparing the data for CSV export
    data_to_export = data[['Image', 'Cost_Range', 'Class_ID']]

    # Exporting to CSV
    export_file_path = 'repair_cost_ranges_merged.csv'  # Update with the desired file path
    data_to_export.to_csv(export_file_path, index=False)

    # Generate the plot with Plotly
    class_frequency = data['Cost_Range'].value_counts().sort_index()
    fig = px.bar(class_frequency, x=class_frequency.index, y=class_frequency.values, 
                 title='Frequency of Different Cost Ranges', labels={'y':'Frequency', 'index':'Cost Range'})

    # Save the plot to a file (this requires Kaleido for static image export)
    plot_file_path = 'cost_range_distribution.png'  # Update with the desired file path
    fig.write_image(plot_file_path)

    # Optionally, show the plot in an interactive window
    fig.show()

    # Print the counts for each bin to verify the distribution
    print(class_frequency)

if __name__ == "__main__":
    main()
