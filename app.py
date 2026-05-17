import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define the model architecture (must match training)
class FairVisionCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(FairVisionCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# App configuration
st.set_page_config(page_title="FairVision - Age Group Classification", page_icon="👀")

st.title("FairVision: Age Group Classification System")
st.markdown("""
This application uses a custom CNN trained on the **FairFace** dataset to predict age groups from face images.
The project focuses on auditing and mitigating bias in AI systems.
""")

# Load the model
@st.cache_resource
def load_model():
    model = FairVisionCNN(num_classes=9)
    # Try to load the best available model (mitigated or baseline)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    mitigated_path = os.path.join(base_dir, "models", "mitigated_model.pth")
    baseline_path = os.path.join(base_dir, "models", "baseline_model.pth")
    
    model_path = mitigated_path if os.path.exists(mitigated_path) else baseline_path
    if not os.path.exists(model_path):
        return None
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

if model is None:
    st.error("Model file not found. Please ensure 'baseline_model.pth' or 'mitigated_model.pth' is in the 'models' directory.")
else:
    st.success("Model loaded successfully.")

    uploaded_file = st.file_uploader("Choose a face image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Preprocessing
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, predicted = torch.max(probabilities, 0)
        
        age_groups = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
        
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Age Group", age_groups[predicted.item()])
        with col2:
            st.metric("Confidence", f"{confidence.item()*100:.2f}%")
            
        # Top 3 predictions
        st.write("---")
        st.write("**Top 3 Predicted Classes:**")
        top3_prob, top3_idx = torch.topk(probabilities, 3)
        for i in range(3):
            st.write(f"{i+1}. {age_groups[top3_idx[i].item()]}: {top3_prob[i].item()*100:.2f}%")

st.sidebar.title("About the Project")
st.sidebar.info("""
**Dataset:** FairFace (balanced across demographics)
**Model:** Custom 3-layer CNN
**Training:** Trained from scratch (no transfer learning)
**Ethical Note:** This is a prototype system. It may exhibit bias despite attempts at mitigation. Use responsibly.
""")
