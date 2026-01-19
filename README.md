**Handwritten Numeral Recognition App (CNN + Capsule Network)**

Handwritten-Numeral-Recog is an intelligent handwritten digit recognition system built using Python, Flask (Backend) and a deep learning model combining Convolutional Neural Networks (CNN) and Capsule Networks (CapsNet).
The system accurately recognizes handwritten digits (0–9) from images and returns the predicted numeral.

**🚀 Key Features**
✅ Handwritten digit recognition (0–9)
✅ Hybrid deep learning model (CNN + Capsule Network)
✅ High accuracy on handwritten inputs
✅ Image preprocessing & normalization
✅ Flask-based prediction API
✅ Simple and efficient architecture

**🏗️ SDLC Overview (Like Building a House)**

This project follows standard SDLC phases:

Planning → Identify the problem of recognizing handwritten digits
Analysis → Study handwritten digit variations and dataset patterns
Design → Design CNN + Capsule Network architecture
Implementation → Model training + Flask backend integration
Testing → Validate accuracy with unseen digit samples
Deployment → Local deployment using Flask (future: cloud-ready)

**🔥 System Architecture (Simple)
**
**❌ Traditional Approach**
Image → Feature Extraction → Classifier → Output
(Limited accuracy due to loss of spatial relationships)
**✅ Proposed Approach (Our System)**
Image → CNN (feature extraction) → Capsule Network (spatial awareness) → Digit Prediction
(Capsules preserve orientation and positional relationships)

**⚙️ Tech Stack**
**Backend**
Python
Flask

NumPy, Pandas

TensorFlow / Keras

Deep Learning

Convolutional Neural Networks (CNN)

Capsule Networks (CapsNet)

Tools

OpenCV (image preprocessing)

Matplotlib (visualization)

📊 Dataset

MNIST Handwritten Digit Dataset

60,000 training images

10,000 testing images

Digits: 0–9

Grayscale images (28×28)

🧠 ML Workflow
Training Phase

Image preprocessing (grayscale, resizing, normalization)

CNN layers extract low-level features

Capsule Network captures spatial relationships

Model trained using labeled digit images

Model artifacts saved for inference.

Prediction Phase

User provides handwritten digit image →
Model processes the image →
System outputs:

Predicted Digit (0–9)

Confidence Score

🎯 Applications

Optical Character Recognition (OCR)

Automated form processing

Bank cheque verification

Postal code recognition

Educational tools

✅ Conclusion

This project demonstrates how combining CNN and Capsule Networks improves handwritten digit recognition by preserving spatial information. The system achieves reliable accuracy and can be extended for real-world OCR applications.
