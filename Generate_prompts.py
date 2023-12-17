import os
import glob
import subprocess
import csv
import re

def is_image_processed(image_name, csv_file):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if image_name in row:
                return True
    return False

def run_command(image_path, prompt):
    print(f"Running command on {image_path}")
    command = [
        "./llava-cli",
        "-m", "/home/hous/Desktop/LLAVA/llama.cpp/models/llava/ggml-model-q4_k.gguf",
        "--mmproj", "/home/hous/Desktop/LLAVA/llama.cpp/models/llava/mmproj-model-f16.gguf",
        "--image", image_path,
        "--temp", "0.1",
        "-p", prompt
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    print("Command output:", result.stdout[:1000])  # Print the first 500 characters of the output for debugging
    return result.stdout

def extract_desired_output(output):
    print("Extracting desired output...")
    # Adjusted pattern to match the new output format
    pattern = r"encode_image_with_clip: image encoded.*?(?=\nllama_model_loader:|$)"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        extracted = match.group(0).strip()
        print("Extracted output:", extracted[:500])  # Print part of the extracted output for debugging
        return extracted
    else:
        print("No match found in the output")
    return "No relevant information found"


def save_result_to_csv(image, result, csv_file):
    with open(csv_file, 'a', newline='') as file:  # Open file in append mode
        writer = csv.writer(file)
        writer.writerow([image, result])

def process_images(folder_path, prompt, csv_file):
    print("Processing images...")
    images = glob.glob(os.path.join(folder_path, '*.jpg')) + glob.glob(os.path.join(folder_path, '*.png'))
    print(f"Found {len(images)} images")

    for image in images:
        image_name = os.path.basename(image)
        if is_image_processed(image_name, csv_file):
            print(f"Image {image_name} is already processed.")
            continue

        output = run_command(image, prompt)
        desired_output = extract_desired_output(output)
        save_result_to_csv(image_name, desired_output, csv_file)

if __name__ == "__main__":
    folder_path = '/home/hous/Desktop/LLAVA/CarDD_release/CarDD_COCO/val2017'
    prompt = "As a mechanical expert, give me the cost to repair this car in euros,straight answer no explanation"
    csv_file = '/home/hous/Desktop/LLAVA/Car_severity_damage_1.csv'

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image", "Result"])

    process_images(folder_path, prompt, csv_file)
    print("Script execution completed.")
