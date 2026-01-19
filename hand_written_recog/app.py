import os
import io
import base64
from PIL import Image
import torch
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for, jsonify
from model import MnistCapsuleModel

app = Flask(__name__)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained Capsule model
model = MnistCapsuleModel().to(device)
model.load_state_dict(torch.load("mnist_capsule1.pt", map_location=device))
model.eval()

# Image transformation (for live input)
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Preprocess base64 canvas image for prediction
def preprocess_image(image_data):
    try:
        image_bytes = base64.b64decode(image_data.split(',')[1])
        image = Image.open(io.BytesIO(image_bytes)).convert('L')
        image = image.resize((128, 128), Image.Resampling.LANCZOS)

        image = transforms.ToTensor()(image)
        image = (image < 0.5).float()  # Invert and binarize

        non_zero = image.squeeze().nonzero(as_tuple=False)
        if non_zero.size(0) == 0:
            return torch.zeros(1, 1, 28, 28).to(device)

        top_left = non_zero.min(0)[0]
        bottom_right = non_zero.max(0)[0]
        bbox = image[:, top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

        bbox = transforms.Resize((20, 20))(bbox)
        padded = torch.zeros(1, 28, 28)
        padded[:, 4:24, 4:24] = bbox
        padded = (padded - 0.1307) / 0.3081

        return padded.unsqueeze(0).to(device)
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        image_data = request.form["imageData"]
        input_tensor = preprocess_image(image_data)

        if input_tensor is None:
            return render_template("result.html", prediction="Invalid Input", image_data=image_data)

        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

        return render_template("result.html", prediction=prediction, image_data=image_data)

    except Exception as e:
        print(f"Prediction failed: {e}")
        return render_template("result.html", prediction="Error", image_data=None)

# Optional: API endpoint for prediction
@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        data = request.json
        image_data = data.get("imageData", "")
        input_tensor = preprocess_image(image_data)

        if input_tensor is None:
            return jsonify({"error": "Invalid image data"}), 400

        output = model(input_tensor)
        prediction = output.argmax(dim=1).item()

        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
