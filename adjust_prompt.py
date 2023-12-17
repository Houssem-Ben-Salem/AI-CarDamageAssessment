import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/home/hous/Desktop/LLAVA/Car_severity_damage_2.csv')

# Function to remove the unwanted text
def clean_text(text):
    return text.split(')', 1)[1].strip() if ')' in text else text

if __name__ == "__main__":
    # Apply the function to the 'Result' column
    df['Result'] = df['Result'].apply(clean_text)
    # Save the modified DataFrame back to a CSV file
    df.to_csv('repair_cost_2.csv', index=False)
