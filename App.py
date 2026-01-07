import pickle
from sklearn.preprocessing import LabelEncoder
from flask import Flask,jsonify,request
import pandas as pd
import numpy as np
model = pickle.load(open("Model_1.pkl","rb"))
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    print("Request received")
    return jsonify({
        "Result": str("Connected!"),
        "Actions": str("Give Input")
    })

@app.route('/', methods=['POST'])
def predict():
    df = pd.read_csv("vegetable_price_weather_demand_season_large.csv")
    df = df.drop(columns='season')

    label_veg = LabelEncoder()
    label_veg.fit_transform(df['vegetable'])
    label_weather = LabelEncoder()
    label_weather.fit_transform(df['weather_condition'])
    label_demand = LabelEncoder()
    label_demand.fit_transform(df['demand_level'])


    in_veg = request.form.get('Vegetable')
    in_weather = request.form.get('Weather')
    in_demand = request.form.get('Demand')

    in_wea_l = label_weather.transform([in_weather])[0]
    in_veg_l = label_veg.transform([in_veg])[0]
    in_dem_l = label_demand.transform([in_demand])[0]
    data = pd.DataFrame([{
        "weather_encoded": in_wea_l,
        "veg_encoded": in_veg_l,
        "demand_encoded": in_dem_l
    }])
    pred = model.predict(data)
    print("Result Sent")
    return jsonify({
        "Predicted Price (Rs. per kg): ": str(round(pred[0],0))
    })
if __name__ == '__main__':
    app.run()