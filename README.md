# AI Car Damage Assessment Tool

## Overview
This repository houses the AI Car Damage Assessment Tool, an advanced solution for analyzing and assessing car damage using artificial intelligence. The tool integrates various AI technologies to deliver accurate and comprehensive car damage assessments, streamlining the process for users ranging from car owners to professionals in the automotive industry.

![image](https://github.com/Houssem-Ben-Salem/AI-CarDamageAssessment/assets/93081419/8806caf6-5845-4887-800d-cdbe7d226745)

![image](https://github.com/Houssem-Ben-Salem/AI-CarDamageAssessment/assets/93081419/fcf107bb-b964-4c85-a2ef-a3bf4df43d2c)


## Features

### AI Realness Check
Ensures the authenticity of car images using sophisticated AI models, providing a reliability check and confidence score for the images analyzed.

### Car Damage Detection
Utilizes advanced object detection algorithms, including YOLO, Mask R-CNN, and Cascade Mask R-CNN, to identify and highlight car damages accurately.

### Damage Severity Assessment
Employs ResNet for feature extraction, enhanced with an attention mechanism, to categorize car damage into different severity levels.

### Cost Prediction
Involves a two-phase approach where CLIP is first trained to match images with damage descriptions. Subsequently, CLIP encodes these matched pairs, which are then fed into a neural network to predict the repair cost range. This method ensures a more accurate and detailed estimation of repair costs, aligning with the specific damages identified in the images.

### LLAVA Integration
Leveraging LLAVA for detailed and accurate damage descriptions.

### Interactive User Interface
Features a user-friendly and intuitive interface, facilitating smooth navigation and interaction for an efficient user experience.

### Comprehensive Damage Report
Generates detailed reports encompassing findings from authenticity checks, severity assessments, and visual damage representations.

## Technical Details

### AI-Generated Image Detection
Adopts a dual-stream architecture that processes images in both frequency and normal domains, enhancing the predictive accuracy of the model.

### Severity Prediction
Utilizes ResNet for extracting features, integrating an attention mechanism for improved accuracy in severity categorization.

### Damage Detection Technologies
Incorporates YOLO, Mask R-CNN, and Cascade Mask R-CNN for precise and detailed damage detection.

### Cost Prediction Methodology
Employs CLIP for initial image and damage description correlation, followed by neural network analysis for cost estimation.

### AI-Generated Damaged Cars Generation
Utilizes a latent consistency model and LLAVA-generated prompts based on an original dataset of damaged cars to generate fake damaged cars.

## Getting Started

### Installation
1. Clone or download this repository.
2. Install the required packages from `requirements.txt`.
3. Download the `llama.cpp` folder from [this link](https://github.com/ggerganov/llama.cpp).
4. Contact me at [houssem.ben.salam@gmail.com](mailto:houssem.ben.salam@gmail.com) to obtain the necessary model weights.

### Usage

Follow the in-app guide to upload images, analyze damage, and generate reports. Check model performance on the dedicated "Models Performance" page within the app.
