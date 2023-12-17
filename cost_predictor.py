import torch
from torch import nn
import clip
from torchvision import transforms

class CarDamageClassifier(nn.Module):
    def __init__(self, clip_model, num_classes, device):
        super(CarDamageClassifier, self).__init__()
        self.clip_model = clip_model

        dummy_image = torch.randn(1, 3, 224, 224, device=device)
        dummy_text = clip.tokenize(["dummy text"]).to(device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(dummy_image)
            text_features = self.clip_model.encode_text(dummy_text)

        image_output_dim = image_features.shape[1]
        text_output_dim = text_features.shape[1]

        self.proj_dim = 1024  # Projection dimension
        self.projector = nn.Linear(image_output_dim + text_output_dim, self.proj_dim)
        
        # Attention mechanism
        self.num_heads = 8  # Number of heads for MultiheadAttention
        self.attention = nn.MultiheadAttention(embed_dim=self.proj_dim, num_heads=self.num_heads, batch_first=True)

        # Define layers with increased complexity and Layer Normalization
        self.fc1 = nn.Linear(self.proj_dim, 2048)
        self.ln1 = nn.LayerNorm(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.ln2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.ln3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 256)
        self.ln4 = nn.LayerNorm(256)
        self.fc5 = nn.Linear(256, num_classes)
        
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout(0.4)

    def forward(self, image, text):
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            text_features = self.clip_model.encode_text(text)

        text_features = text_features.to(image_features.dtype)

        combined_features = torch.cat((image_features, text_features), dim=1)
        combined_features = combined_features.to(dtype=self.projector.weight.dtype)

        projected_features = self.projector(combined_features)

        # Apply attention
        batch_size = image.size(0)
        attn_output, _ = self.attention(projected_features.unsqueeze(1), projected_features.unsqueeze(1), projected_features.unsqueeze(1))
        attn_output = attn_output.reshape(batch_size, -1)

        # Pass through the network with Layer Normalization and dropout layers
        x = self.fc1(attn_output)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        x = self.ln4(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc5(x)
        return x

def load_clip_model(model_path, device):
    # Load the fine-tuned CLIP model
    clip_model, _ = clip.load("RN50", device=device, jit=False)
    clip_model.load_state_dict(torch.load(model_path))
    clip_model.eval()
    return clip_model

def load_cost_model(model_path, clip_model, device):
    model = CarDamageClassifier(clip_model, num_classes=4, device=device)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Image transformation
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_cost(image, text, model, device):
    if model is None:
        raise ValueError("Model is None. Ensure the model is properly loaded.")
    
    # Define the class names
    class_names = ["More than 10000 euros","Between 1000 euros and 3000 euros","Less than 1000 euros","between 3000 euros and 10000 euros"]
    
    # Transform the image
    image = val_transforms(image).unsqueeze(0).to(device)

    # Ensure the model is in evaluation mode
    model.eval()

    # Process inputs
    with torch.no_grad():
        text_input = clip.tokenize([text]).to(device)

        # Get the prediction
        outputs = model(image, text_input)
        _, predicted = torch.max(outputs, 1)

        # Map prediction to class name
        predicted_cost_range = class_names[predicted.item()]
    
    return predicted_cost_range

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_model = load_clip_model('/home/hous/Desktop/LLAVA/best_matching_model.pth', device)
    cost_model = load_cost_model('/home/hous/Desktop/LLAVA/best_cost_prediction_model.pth', clip_model, device)
    print("Model loaded and ready for predictions")