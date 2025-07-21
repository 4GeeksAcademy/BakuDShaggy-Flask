
from flask import Flask, render_template, request
import pickle
import os
# your code here

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Create the Flask app with custom template folder
app = Flask(__name__, template_folder=os.path.join(current_dir, '../templates'))

# Load the model
model_path = os.path.join(current_dir, 'iris_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Define a dictionary for class names
class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the values from the form
        val1 = float(request.form['sepal_length'])
        val2 = float(request.form['sepal_width'])
        val3 = float(request.form['petal_length'])
        val4 = float(request.form['petal_width'])
        
        data = [[val1, val2, val3, val4]]
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        pred_class = None
    
    return render_template('index.html', prediction=pred_class)

if __name__ == '__main__':
    app.run(debug=True)