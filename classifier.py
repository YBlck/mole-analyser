import os

from fastai.vision.core import PILImage

LABEL = ["Safe", "Dangerous"]


def classify(model, filepath):
    try:
        img = PILImage.create(filepath)
        pred, pred_idx, probs = model.predict(img)
        prob = float(probs[pred_idx]) * 100
        result = {
            "prediction": LABEL[int(pred)],
            "probability": round(prob, 2)
        }
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

    return result
