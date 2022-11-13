import json
from xgboost import XGBClassifier


def make_prediction(x):
    # Load the saved model
    loaded_model = XGBClassifier()
    loaded_model.load_model("main_model.model")
    predictions_out = loaded_model.predict(x)
    print(predictions_out)
    # Make prediction
    dict_out = {}
    for count, value in enumerate(predictions_out):
        dict_out[count] = float(value)

    # Load json for decoding and decode the output
    with open('encoder.json') as json_file:
        data = json.load(json_file)

    # Returns the actual species name
    return data[str(int(dict_out[0]))]
