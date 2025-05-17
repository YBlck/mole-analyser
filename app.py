import os

import gdown
from fastai.vision.all import load_learner
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

from classifier import classify

app = Flask(__name__)

STATIC_FOLDER = "static"
UPLOAD_FOLDER = STATIC_FOLDER + "/uploads/"
MODEL_FILE_NAME = "mole_checker_model.pkl"
MODEL_PATH = STATIC_FOLDER + "/model/" + MODEL_FILE_NAME
MODEL_URL = "https://drive.google.com/uc?id=16V2HmnBnCCJMInGQNrc0EU0A_AZWZb1f"

if not os.path.exists(MODEL_PATH):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

cnn_model = load_learner(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            result = classify(cnn_model, filepath)

            return render_template(
                "index.html",
                result=result["prediction"],
                prob=result["probability"],
            )

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run()
