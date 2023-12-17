import streamlit as st
import ai_detector_model
import car_damage_detector
from car_damage_cascade import setup_cfg_cascade, load_model_cascade, detect_damage_cascade 
from detectron2.data import DatasetCatalog, MetadataCatalog
import cv2
import time
import base64
import pdfkit
from build_report import convert_image_to_base64, create_html_report
import plotly.graph_objs as go
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import severity_detector
import asyncio
import tempfile
import os
import cost_predictor
from ultralytics import YOLO
from PIL import Image
import torch
from detectron2.data.datasets import register_coco_instances


dataset_name = "my_dataset_val"
# Load the dataset
dataset_dicts = DatasetCatalog.get(dataset_name)

severity_model_path = '/home/hous/Desktop/LLAVA/best_model_weights.pth' 
severity_model = severity_detector.SeverityDetector(severity_model_path)
yolo_model = YOLO('/home/hous/Desktop/LLAVA/Yolo_best_model.pt')

# Path to the wkhtmltopdf executable
path_wkhtmltopdf = '/usr/bin/wkhtmltopdf'

# Create a configuration object for pdfkit with the specified wkhtmltopdf path
config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)

# Load the fine-tuned CLIP model and cost prediction model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = cost_predictor.load_clip_model('/home/hous/Desktop/LLAVA/best_matching_model.pth', "cpu")
cost_model = cost_predictor.load_cost_model('/home/hous/Desktop/LLAVA/best_cost_prediction_model.pth', clip_model, "cpu")
if cost_model is None:
    st.error("Failed to load the cost prediction model.")
# Retrieve metadata
dataset_metadata = MetadataCatalog.get(dataset_name)
# Load your pre-trained models
ai_model_path = '/home/hous/Desktop/LLAVA/best_combined_resnet_model.pth'
ai_model, ai_device = ai_detector_model.load_model(ai_model_path)

car_damage_model_path = '/home/hous/Desktop/LLAVA/output1/model_final.pth' 
car_damage_model = car_damage_detector.load_model(car_damage_model_path)

car_damage_model_cascade_path = '/home/hous/Desktop/LLAVA/model_0002499.pth' 
car_damage_cascade_model = load_model_cascade(car_damage_model_cascade_path)

async def run_llava_cli_async(image_path, prompt):
    command = [
        "./llava-cli",
        "-m", "/home/hous/Desktop/LLAVA/llama.cpp/models/llava/ggml-model-q4_k.gguf",
        "--mmproj", "/home/hous/Desktop/LLAVA/llama.cpp/models/llava/mmproj-model-f16.gguf",
        "--image", image_path,
        "--temp", "0.1",
        "-p", prompt
    ]
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await process.communicate()
    if process.returncode == 0:
        return stdout.decode()
    else:
        return f"Error: {stderr.decode()}"

def run_llava_cli(image_path, prompt):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(run_llava_cli_async(image_path, prompt))
    loop.close()
    return result

# Function to remove the unwanted text
def clean_text(text):
    return text.split(')', 1)[1].strip() if ')' in text else text

def predict_with_yolo(image_np, confidence_threshold):
    # Directly use the NumPy array for prediction
    results = yolo_model.predict(image_np, conf=confidence_threshold)

    # Extract boxes and plot the results
    boxes = results[0].boxes
    res_plotted = results[0].plot()[:, :, ::-1]

    return boxes, res_plotted

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg", dir="/home/hous/Desktop/LLAVA/temp") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None
      
def display_report(image, prediction, severity, damages, llava_result,report_image):
    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown("### Report Summary")
        st.markdown(f"**Image Assessment:** {'AI-Generated' if prediction == 'AI-Generated' else 'Real'}")
        st.markdown(f"**Damage Severity:** {severity}")
        st.markdown(f"**Detailed Description:** {llava_result}")
        st.markdown("### Damage Details")
        for damage_detail in damages:
            st.markdown(f"- {damage_detail}")

    with col2:
        st.markdown("### Visual Inspection")
        st.image(report_image, caption='Processed Image', use_column_width=True)


def create_download_link(filename):
    with open(filename, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    href = f'<a href="data:file/pdf;base64,{base64_pdf}" download="{os.path.basename(filename)}">Download Report as PDF</a>'
    st.markdown(href, unsafe_allow_html=True)


def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def main():
    st.set_page_config(page_title='AI Car Damage Assessment', layout='wide')
    local_css("/home/hous/Desktop/LLAVA/style.css") 
    st.title("AI Car Damage Assessment")

    # Sidebar for navigation
    with st.sidebar:
        st.info("Navigate through the app:")
        page = st.radio("Choose a page:", ["INFO", "Explore Features", "Models Performance"])

    # Page navigation
    if page == "Explore Features":
        display_home_page()  # Function that contains your main app logic
    elif page == "INFO":
        display_app_features_page()  # Function that explains app features
    elif page == "Models Performance":
        display_model_performance_page()

def display_home_page():
    temp_image_path = None
    with st.sidebar:
        st.info("Instructions: Upload an image of the car to check if it's real and detect any damage.")

    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
    
    if uploaded_image is not None:
        if 'uploaded_image' not in st.session_state or st.session_state['uploaded_image'] != uploaded_image:
            st.session_state['uploaded_image'] = uploaded_image
            st.session_state['image_processed'] = False

    if uploaded_image is not None and not st.session_state.get('image_processed', False):
        st.session_state['image_processed'] = True  
        temp_image_path = save_uploaded_file(uploaded_image)
    
        if temp_image_path and 'llava_task' not in st.session_state:
            prompt = "What damages does this car have, i need a detailed report with possible solutions and advices."
            with ThreadPoolExecutor() as executor:
                future = executor.submit(run_llava_cli, temp_image_path, prompt)
                st.session_state['llava_task'] = future
    
    if 'uploaded_image' in st.session_state and st.session_state['uploaded_image'] is not None:
            pil_image = Image.open(st.session_state['uploaded_image']).convert('RGB')
            st.markdown('image not displaying')
            st.image(pil_image, caption='Uploaded Image', use_column_width=True)
    # AI Check
    if st.button('Check if Image is Real or AI-Generated'):
        start_time = time.time()
        with st.spinner('Processing...'):
            prediction, confidence = ai_detector_model.predict(pil_image, ai_model, ai_device)
        end_time = time.time()
        processing_time = end_time - start_time
        st.success(f"Done in {processing_time:.2f}s")
        st.markdown(f"**Prediction:** The image is **{prediction}** with a confidence of **{confidence:.2f}%**.")
        st.session_state['prediction'] = prediction  # Save prediction to session state
        torch.cuda.empty_cache()
    # Damage Detection
    if 'prediction' in st.session_state and st.session_state['prediction'] == 'Real':
        model_choice = st.selectbox("Select Prediction Model", ["Cascade Mask R-CNN","Mask-RCNN", "YOLO"])
        if st.button('Detect Car Damage'):
            start_time = time.time()
            with st.spinner('Detecting damage...'):
                open_cv_image = np.array(pil_image)[:, :, ::-1]

                # Choose model based on selection
                if model_choice == "YOLO":
                    # Call YOLO prediction function
                    boxes, res_plotted = predict_with_yolo(open_cv_image, confidence_threshold=0.5)
                    st.session_state['processed_image'] = res_plotted
                    # Display YOLO results
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)
                elif model_choice == "Mask-RCNN" :
                    # Mask-RCNN
                    processed_image = car_damage_detector.detect_damage(open_cv_image, car_damage_model, dataset_metadata)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.session_state['damages'] = []  # Initialize the list of damages
                    outputs = car_damage_model(open_cv_image)  # Get model outputs
                    instances = outputs["instances"]
                    if instances.has("pred_boxes"):
                        boxes = instances.pred_boxes.tensor.cpu().numpy()
                        classes = instances.pred_classes.cpu().numpy()
                        for i in range(len(boxes)):
                            class_name = dataset_metadata.thing_classes[classes[i]]
                            st.session_state['damages'].append(class_name)
                        st.image(processed_image, caption='Car Damage Detection', use_column_width=True)
                        st.session_state['processed_image'] = processed_image
                    else:
                        st.session_state['damages'].append('No damages detected')

                else:
                    # Existing code for Mask-RCNN
                    processed_image = detect_damage_cascade(open_cv_image, car_damage_cascade_model, dataset_metadata)
                    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                    st.session_state['damages'] = []  # Initialize the list of damages
                    outputs = car_damage_cascade_model(open_cv_image)  # Get model outputs
                    instances = outputs["instances"]
                    if instances.has("pred_boxes"):
                        boxes = instances.pred_boxes.tensor.cpu().numpy()
                        classes = instances.pred_classes.cpu().numpy()
                        for i in range(len(boxes)):
                            class_name = dataset_metadata.thing_classes[classes[i]]
                            st.session_state['damages'].append(class_name)
                        st.image(processed_image, caption='Car Damage Detection', use_column_width=True)
                        st.session_state['processed_image'] = processed_image
                    else:
                        st.session_state['damages'].append('No damages detected')
            processing_time = time.time() - start_time
            st.success(f"Damage detection done in {processing_time:.2f}s")
            torch.cuda.empty_cache()

    
        # Severity Assessment
        if st.button('Determine Severity of Damage'):
            start_time = time.time()
            with st.spinner('Assessing damage severity...'):
                severity = severity_model.predict(pil_image)
            end_time = time.time()
            processing_time = end_time - start_time
            st.success(f"Severity assessment done in {processing_time:.2f}s")
            st.markdown(f"**Severity of Damage:** {severity}")
            st.session_state['severity'] = severity
            torch.cuda.empty_cache()

        if st.button('Predict Cost'):
            start_time = time.time()
            with st.spinner('Predicting cost...'):
                # Retrieve the uploaded image from the session state
                uploaded_image = st.session_state.get('uploaded_image', None)
                if uploaded_image is not None:
                    pil_image = Image.open(uploaded_image)

                    # Create the text input from the detected damages
                    damages = st.session_state.get('damages', [])
                    text_input = "This car has " + ', '.join(damages)

                    # Predict the cost
                    predicted_cost_range = cost_predictor.predict_cost(pil_image, text_input, cost_model, device)
                    end_time = time.time()
                    # Display the result
                    st.markdown(f"**Predicted Cost Range:** {predicted_cost_range}")
                    torch.cuda.empty_cache()
                else:
                    st.error("No image uploaded for cost prediction.")

        
        if st.button('Generate Report'):
            # You would need to retrieve the prediction, severity, and damages list
            # Here's an example of how you might get this data
            prediction = st.session_state.get('prediction', 'Unknown')
            severity = st.session_state.get('severity', 'Unknown')
            processed_image = st.session_state.get('processed_image', ['No damages detected'])
            damages = st.session_state.get('damages', ['No damages detected'])
            if 'llava_task' in st.session_state:
                my_bar = st.progress(0)
                progress_text = st.empty()
                for percent_complete in range(100):
                    time.sleep(3.5)
                    my_bar.progress(percent_complete + 1)
                    progress_text.text(f"Now at {percent_complete + 1}% Hang tight, we're faster than a snail on a skateboard!")
                raw_llava_result = st.session_state['llava_task'].result()
                cleaned_llava_result = clean_text(raw_llava_result)
            progress_text.empty()
            torch.cuda.empty_cache()
            # Call the report display function
            display_report(pil_image, prediction, severity, damages,cleaned_llava_result,processed_image)


            # Convert report data to HTML
            html_report = create_html_report(prediction, severity, damages, cleaned_llava_result, processed_image)

            # Specify the path for the temporary PDF file
            pdf_file_path = 'report.pdf'

            # Generate PDF from HTML using the configured path for wkhtmltopdf
            pdfkit.from_string(html_report, pdf_file_path, configuration=config)

            # Create a download link for the PDF
            create_download_link(pdf_file_path)

            if temp_image_path and os.path.exists(temp_image_path):
                os.remove(temp_image_path)

def display_app_features_page():
    st.markdown("""
        ### 🌟 Welcome to the AI Car Damage Assessment Tool! 🌟
        Welcome to the future of car damage assessment! Our tool combines cutting-edge AI technology with user-friendly design to revolutionize how you approach car damage analysis.

        ---

        ### 🤖 AI Realness Check
        - **What it Does**: Verifies the authenticity of your car's image using a sophisticated AI model. 
        - **Why It's Awesome**: Provides a confidence score to assure you're working with real images, setting a reliable foundation for further analysis. 

        ---

        ### 🛠️ Car Damage Detection
        - **What it Does**: Employs state-of-the-art object detection to spot and highlight damages on your car. 
        - **Why It's Awesome**: Offers a visual map of all damage types, making it easier to understand the extent of repairs needed. 

        ---

        ### 📊 Damage Severity Assessment
        - **What it Does**: Evaluates how severe the car damage is, categorizing it into different levels. 
        - **Why It's Awesome**: Helps in making informed decisions by understanding the damage severity. 

        ---

        ### 💲 Cost Prediction
        - **What it Does**: Estimates repair costs based on damage type and severity. 
        - **Why It's Awesome**: Aids in financial planning and insurance claims with AI-driven cost range predictions. 

        ---

        ### 💻 LLAVA CLI Integration
        - **What it Does**: Processes images via a command-line interface for detailed damage descriptions. 
        - **Why It's Awesome**: Delivers expert-level insights into your car’s condition, straight from the AI's brain. 

        ---

        ### 👩‍💻 Interactive User Interface
        - **What it Does**: Offers an intuitive interface with easy navigation and interactive elements. 
        - **Why It's Awesome**: Ensures a smooth and hassle-free user experience, from uploading images to viewing reports. 

        ---

        ### 📋 Comprehensive Damage Report
        - **What it Does**: Compiles a detailed report of all findings including authenticity checks, severity, and visual damage representation.
        - **Why It's Awesome**: Provides a complete overview of the car's condition for your records or to share with others.
        - **Emoji Summary**: 📄🔍🚗

        ---

        ### Get Started Now! 🚀
        Upload your car's image and let our AI magic take care of the rest! 
    """, unsafe_allow_html=True)
    
def display_model_performance_page():
    st.header("Model Performance")

    # Function to display performance metrics for a given feature
    def display_feature_performance(feature_name, models, metrics):
        st.subheader(f"{feature_name} Performance Metrics")

        # Dropdown for selecting the metric
        metric_name = st.selectbox(f"Select metric for {feature_name}", list(metrics.keys()))

        # Data for the selected metric
        metric_values = metrics[metric_name]

        # Create a line chart with markers for the selected metric
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=models, y=metric_values, mode='markers+lines', name=metric_name))

        fig.update_layout(title=f"{metric_name} Comparison for {feature_name}",
                          xaxis_title="Models",
                          yaxis_title=metric_name,
                          yaxis=dict(range=[0, 1]))  # Adjust y-axis range if needed
        st.plotly_chart(fig)

    # Example metrics for multiclass problems (placeholder values)
    # Feature 1: Fake/Real Image Detector
    feature_1_models = ["Model 1", "Model 2", "Model 3"]
    feature_1_metrics = {
        "Accuracy": [0.90, 0.92, 0.88],
        "F1 Score": [0.89, 0.91, 0.87],
        "Recall": [0.88, 0.90, 0.86],
        "Precision": [0.87, 0.89, 0.85]
    }
    display_feature_performance("Fake/Real Image Detector", feature_1_models, feature_1_metrics)

    # Feature 2: Severity Assessment
    severity_models = ["Severity Model 1", "Severity Model 2"]
    severity_metrics = {
        "Weighted F1 Score": [0.75, 0.78],
        "Weighted Precision": [0.76, 0.79],
        "Weighted Recall": [0.74, 0.77]
    }
    display_feature_performance("Severity Assessment", severity_models, severity_metrics)

    # Feature 3: Cost Range Prediction
    cost_models = ["Cost Model A", "Cost Model B"]
    cost_metrics = {
        "Weighted F1 Score": [0.65, 0.68],
        "Weighted Precision": [0.66, 0.69],
        "Weighted Recall": [0.64, 0.67]
    }
    display_feature_performance("Cost Range Prediction", cost_models, cost_metrics)

    def display_detection_performance(feature_name, models, metrics):
        st.subheader(f"{feature_name} Performance Metrics")

        # Dropdown for selecting the metric type (bbox or seg)
        metric_type = st.selectbox(f"Select metric type for {feature_name}", ['bbox', 'seg'])

        # Dropdown for selecting the specific metric
        metric_name = st.selectbox(f"Select specific metric for {feature_name} ({metric_type})", list(metrics[metric_type].keys()))

        # Data for the selected metric
        metric_values = metrics[metric_type][metric_name]

        # Create a line chart with markers for the selected metric
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=models, y=metric_values, mode='markers+lines', name=metric_name))

        fig.update_layout(title=f"{metric_name} Comparison for {feature_name} ({metric_type})",
                          xaxis_title="Models",
                          yaxis_title=metric_name,
                          yaxis=dict(range=[0, 1]))  # Adjust y-axis range if needed
        st.plotly_chart(fig)

    # Placeholder values for Damage Detection feature
    damage_detection_models = ["DD Model 1", "DD Model 2", "DD Model 3"]
    damage_detection_metrics = {
        'bbox': {
            "AP": [0.45, 0.48, 0.50],
            "AP50": [0.65, 0.68, 0.70],
            "AP75": [0.40, 0.43, 0.45],
            "mAP": [0.50, 0.53, 0.55]
        },
        'seg': {
            "AP": [0.46, 0.49, 0.51],
            "AP50": [0.66, 0.69, 0.71],
            "AP75": [0.41, 0.44, 0.46],
            "mAP": [0.51, 0.54, 0.56]
        }
    }
    display_detection_performance("Damage Detection", damage_detection_models, damage_detection_metrics)

if __name__ == "__main__":
    print("Starting Streamlit App")
    main()
