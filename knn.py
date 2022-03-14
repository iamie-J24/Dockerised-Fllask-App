import pandas as pd
import numpy as np
from flask import Flask, render_template, request
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
app  = Flask(__name__)

@app.route("/")

def home():
    df = pd.read_csv('cars.csv')
    data = df.drop(['class'], axis = 1)
    target = df['class']
    data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.20, random_state = 10)
    #create object of the lassifier
    neigh = KNeighborsClassifier(n_neighbors=3)
    #Train the algorithm
    neigh.fit(data_train, target_train)

    # Save model in pickle file
    pickle.dump(neigh,open("cars.pkl", "wb"))
    # predict the response
    pred = neigh.predict(data_test)
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    buying = request.form['buying']
    maintainance = request.form['maint']
    doors = request.form['doors']
    persons = request.form['persons']
    lug_boot = request.form['lug_boot']
    safety = request.form['safety']

    form_array = np.array([[buying, maintainance, doors, persons, lug_boot, safety]])
    model = pickle.load(open("cars.pkl", "rb"))

    pred = model.predict(form_array)[0]
    if pred == 0:
        result = "Acceptable Class"
    elif pred == 1:
        result = "Good Class"
    elif pred == 2:
        result = "Unacceptable Class"
    else:
        result = "Very Good Class"

    return render_template("pred.html", result = result)

if __name__ == "__main__":
    app.run(debug=True, port=5001, host="0.0.0.0")