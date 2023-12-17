import json
import csv

def json_to_csv(json_file_path, csv_file_path):
    # Load the JSON data
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)

    # Mapping category IDs to names
    category_map = {category['id']: category['name'] for category in json_data['categories']}

    # Creating a dictionary for damages for each image
    image_damages = {}
    for annotation in json_data['annotations']:
        image_id = annotation['image_id']
        damage = category_map[annotation['category_id']]
        if image_id in image_damages:
            image_damages[image_id].add(damage)
        else:
            image_damages[image_id] = {damage}

    # Mapping image IDs to file names
    image_id_to_file_name = {image['id']: image['file_name'] for image in json_data['images']}

    # Preparing the data for the CSV file
    csv_data = []
    for image_id, damages in image_damages.items():
        file_name = image_id_to_file_name.get(image_id, "Unknown")
        damage_sentence = "The car has " + ', '.join(damages)
        csv_data.append({'file_name': file_name, 'damages': damage_sentence})

    # Writing the data to a CSV file
    with open(csv_file_path, mode='w', newline='') as outfile:
        fieldnames = ['file_name', 'damages']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)

        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)

json_to_csv('/home/hous/Desktop/LLAVA/CarDD_release/CarDD_COCO/annotations/instances_test2017.json', 'damage_image_pair_test.csv')
