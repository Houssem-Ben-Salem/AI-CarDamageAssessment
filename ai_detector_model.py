import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# Define your model classes (FFTResNet and CombinedResNet) here
class FFTResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(FFTResNet, self).__init__()
        # Load a pretrained ResNet50 model
        self.resnet50 = models.resnet50(pretrained=pretrained)
        # Remove the last fully connected layer to get feature vector
        self.resnet50 = nn.Sequential(*(list(self.resnet50.children())[:-1]))

    def forward(self, x):
        # Apply FFT
        x = self.apply_fft(x)
        # Feature extraction with ResNet50
        features = self.resnet50(x)
        # Flatten the features
        return features.view(features.size(0), -1)

    def apply_fft(self, x):
        # Applying FFT
        x = torch.fft.fft2(torch.tensor(x, dtype=torch.complex64))
        # Shifting the zero frequency component to the center
        x = torch.fft.fftshift(x)
        # Taking the magnitude (abs) and applying log scale
        x = torch.log(torch.abs(x) + 1)
        # Normalizing and returning the real part
        return x.real

class CombinedResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(CombinedResNet, self).__init__()

        # Standard ResNet50
        self.resnet50_standard = models.resnet50(pretrained=pretrained)
        self.resnet50_standard = nn.Sequential(*(list(self.resnet50_standard.children())[:-1]))

        # FFTResNet
        self.fft_resnet = FFTResNet(pretrained=pretrained)

        # Assuming both ResNet50 and FFTResNet output 2048-dimensional features
        combined_features_size = 2048 * 2  # Change accordingly if different

        # Fully connected layers for classification
        self.classifier = nn.Sequential(
            nn.Linear(combined_features_size, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)  
        )

    def forward(self, x):
        # Extract features from standard ResNet50
        features_standard = self.resnet50_standard(x).view(x.size(0), -1)

        # Extract features from FFTResNet
        features_fft = self.fft_resnet(x)

        # Concatenate the features from both streams
        combined_features = torch.cat((features_standard, features_fft), dim=1)

        # Classify using the combined features
        output = self.classifier(combined_features)
        return output

def load_model(model_path):
    """
    Load the pre-trained CombinedResNet model.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CombinedResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set the model to evaluation mode
    return model, device

def predict(image, model, device):
    """
    Make a prediction on a single image using the pre-trained model.
    """
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Define the transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transform the image
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Make the prediction
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        top_prob, top_catid = torch.max(probabilities, 1)

    # Convert the prediction to a human-readable format
    classes = ['AI-Generated','Real']
    prediction = classes[top_catid.item()]
    confidence = top_prob.item() * 100  # Convert to percentage

    return prediction, confidence