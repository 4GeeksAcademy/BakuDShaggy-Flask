from utils import db_connect
engine = db_connect()
from flask import Flask, render_template, request
import pickle
import numpy as np
# your code here
app = Flask(__name__)

# Load model
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Species mapping
species = {
    0: "Iris Setosa",
    1: "Iris Versicolor",
    2: "Iris Virginica"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Get form values
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]
        # Predict
        result = model.predict([features])[0]
        prediction = species[result]
        
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)