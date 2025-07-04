# 🧪 Mole Analyser

**Mole Analyser** is a lightweight web application built with **Flask** and powered by a **FastAI deep learning model**. It classifies skin mole images as either *safe* or *suspicious* based on visual features.

![App Screenshot](static/images/preview.png)

---

## 🚀 Features

- 📸 Upload a skin mole image
- 🤖 Analyze it using a trained convolutional neural network
- 📊 Get a prediction and confidence score
- 💻 Simple web interface (Bootstrap 5)
- ☁️ Lazy model download from Google Drive

---

## 🧠 Model Info

- Trained on: [HAM10000 dataset](https://www.kaggle.com/datasets/surajghuwalewala/ham1000-segmentation-and-classification)
- Architecture: `resnet34`
- Accuracy: ~90%
- Size: ~83MB (auto-downloaded on first run)
- ML project: [Google colab](https://colab.research.google.com/drive/1sHKj0PslExX3J_V0wChRxqDz_GXk4hxO?usp=sharing)

---

## 📁 Project Structure
```text
mole-analyser/
├── app.py                # Main Flask application
├── classifier.py         # Prediction logic
├── requirements.txt      # Python dependencies
├── static/
│   ├── model/            # Directory for downloaded model
│   ├── uploads/          # Uploaded image storage
│   └── images/           # App assets (e.g., screenshot)
└── templates/
    └── index.html        # Main UI template
```

---

## 📦 Installation

```bash
git clone https://github.com/YBlck/mole-analyser.git
cd mole-analyser
pip install -r requirements.txt
```

---

## ▶️ Run the app
```bash
python app.py
```
The application will be available at:
http://localhost:5000

On first run, the model (~83MB) will be automatically downloaded from Google Drive.

---

## ⚠️ Disclaimer

**Warning:** This project was developed for educational purposes only. It is not a medical or diagnostic tool. Do not rely on its output to make health-related decisions. Always consult a qualified healthcare professional for any medical concerns.

