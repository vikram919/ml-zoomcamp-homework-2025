import json
from io import BytesIO
from urllib import request

import numpy as np
import onnxruntime as ort
from PIL import Image
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

onnx_model_path = 'hair_classifier_empty.onnx'
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
target_size = (200,200)
IMG_MEAN =  np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD =  np.array([0.485, 0.456, 0.406], dtype=np.float32)

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def preprocess(img):
  resized_img = prepare_image(img, target_size)
  x = np.array(resized_img).astype("float32") / 255.0
  x = (x - IMG_MEAN[None, None, :]) / IMG_STD[None, None, :]
  x = np.transpose(x, (2, 0, 1))  # CHW
  x = np.expand_dims(x, axis=0)  # NCHW
  return x


def predict(url):
    img = download_image(url)
    X = preprocess(img)

    preds = session.run(
        output_names=[output_name],
        input_feed={input_name:X }
    )

    # What happens here is we take an Numpy array and
    # it will be converted to usual python floats.
    float_predictions = float(preds[0].squeeze())

    return float_predictions

def lambda_handler(event, context):
    logger.info(f"request received to process for event: {event}")
    url = event.get("url")
    if not url:
        return {"statusCode": 400, "body": json.dumps({"error": "Missing url"})}
    result = predict(url)
    return {
        "statusCode": 200,
        "body": json.dumps({"prediction": result})
    }