from PIL import Image
from io import BytesIO
import base64
import numpy as np 

def convert_image_to_base64(image):
    # Check if the image is a NumPy array and convert it to a PIL Image if it is
    if isinstance(image, np.ndarray):
        # Convert NumPy array to PIL Image
        image = Image.fromarray(image)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_html_report(prediction, severity, damages, llava_result, processed_image):
    # Convert the processed image to base64 for embedding in HTML
    processed_image_base64 = convert_image_to_base64(processed_image)

    # Create a structured HTML report
    html_content = f"""
    <html>
    <head>
    <title>Car Damage Assessment Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f4f4f4;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            width: 80%;
            margin: auto;
            overflow: hidden;
        }}
        .header {{
            background: #333;
            color: #fff;
            padding: 20px 0;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
        }}
        .section {{
            margin-bottom: 20px;
        }}
        .section-title {{
            background: #e3e3e3;
            padding: 10px;
            font-size: 20px;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        table, th, td {{
            border: 1px solid #ccc;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
        }}
    </style>
    </head>
    <body>
    <div class="container">
        <div class="header">
            <h1>Car Damage Assessment Report</h1>
        </div>
        <div class="section">
            <div class="section-title">Ai-Generated image or real : </div> <p>{prediction}</p>
        </div>
        <div class="section">
            <div class="section-title">Damage Severity : </div> <p>{severity}</p>
        </div>
        <div class="section">
            <div class="section-title">Damages Detected</div>
            <ul>
                {''.join(f'<li>{damage}</li>' for damage in damages)}
            </ul>
        </div>
        <div class="section">
            <div class="section-title">Detailed Description</div>
            <p>{llava_result}</p>
        </div>
        <div class="section">
            <div class="section-title">Processed Image</div>
            <div class="image-container">
                <img src="data:image/jpeg;base64,{processed_image_base64}">
            </div>
        </div>
    </div>
    </body>
    </html>
    """
    return html_content