import os

from fastai.vision.core import PILImage

LABEL = ["Safe", "Suspicious"]


def classify(model, filepath):
    try:
        img = PILImage.create(filepath)
        prediction, prediction_idx, probs = model.predict(img)
        prob = float(probs[prediction_idx]) * 100
        result = {
            "prediction": LABEL[int(prediction_idx)],
            "probability": round(prob, 2)
        }
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return result
