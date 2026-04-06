import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms, models

# -------------------------------
# CONFIG
# -------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 224

# -------------------------------
# LOAD MODEL
# -------------------------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("best_resnet18_ksd.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# -------------------------------
# PREPROCESS
# -------------------------------
def apply_clahe(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(enhanced)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Lambda(lambda x: apply_clahe(x)),
    transforms.ToTensor()
])

# -------------------------------
# GRAD-CAM
# -------------------------------
class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.layer = layer
        self.gradients = None
        self.activations = None

        layer.register_forward_hook(lambda m, i, o: setattr(self, "activations", o))
        layer.register_full_backward_hook(lambda m, gi, go: setattr(self, "gradients", go[0]))

    def generate(self, x, class_idx):
        out = self.model(x)
        self.model.zero_grad()
        out[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2,3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = torch.relu(cam)
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().detach().numpy()

# Use deeper layer for better results
gradcam = GradCAM(model, model.layer4)

# -------------------------------
# STREAMLIT UI
# -------------------------------
st.title("Kidney Stone Detection 🩺")

uploaded_file = st.file_uploader("Upload CT Scan", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs).item()
        conf = probs[0][pred].item()

    label = "Stone" if pred == 1 else "Normal"
    st.subheader(f"Prediction: {label} ({conf*100:.2f}%)")

    # -------------------------------
    # GRAD-CAM ONLY
    # -------------------------------
    if pred == 1:
        st.write("### 🔥 Grad-CAM Analysis")

        cam = gradcam.generate(input_tensor, pred)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        # Normalize
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        # Heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Original image
        orig = np.array(image.resize((IMG_SIZE, IMG_SIZE))) / 255.0

        # Overlay
        overlay = heatmap / 255.0 * 0.4 + orig * 0.6
        overlay = (overlay - overlay.min()) / (overlay.max() + 1e-8)

        st.image(overlay, caption="Grad-CAM Overlay")

    else:
        st.success("No kidney stone detected ✅")