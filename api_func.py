from fastapi import FastAPI
import numpy as np
from make_pred import make_prediction
from train_model import make_model_save, add_to_data

app = FastAPI()


@app.get("/{x1}/{x2}/{x3}/{x4}")
def get_pred(x1: float, x2: float, x3: float, x4: float):
    p1 = [x1, x2, x3, x4]
    x = np.array([p1])

    dict_out = make_prediction(x)
    return dict_out


@app.get("/train_model")
def train_model():
    make_model_save()
    return {"Response": "Training completed."}


@app.get("/{x1}/{x2}/{x3}/{x4}/{species}")
def add_to_data_api(x1: float, x2: float, x3: float, x4: float, species: str):
    add_to_data(x1, x2, x3, x4, species)
