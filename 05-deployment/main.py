import os
import pickle
from fastapi import FastAPI

model_file_name = "pipeline_v1.bin"

def load_model(model_path: str):
    with open(model_path, "rb") as f_in:
        dict_vectorizer, model = pickle.load(f_in)
    return dict_vectorizer, model

dict_vectorizer, model = load_model(os.path.join(os.getcwd(), model_file_name))

    
app = FastAPI()

def guess_if_student_is_converted(student_data: dict):
    vectorized_data = dict_vectorizer.transform([student_data])
    conversion_probability = model.predict_proba(vectorized_data)[:, 1]
    if conversion_probability >= 0.5:
        return {"conversion_probability": float(conversion_probability), "converted": True}
    return {"conversion_probability": float(conversion_probability), "converted": False}

@app.post("/ai")
async def do_ai(request_data: dict = {}):
    return guess_if_student_is_converted(request_data)

@app.get("/")
async def root():
    return {"message": "Hello World!"}