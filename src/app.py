from flask import Flask, render_template, request
import pickle
import os

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Correct template path - templates are in parent directory
template_path = os.path.join(current_dir, '..', 'templates')

# Create Flask app with custom template folder
app = Flask(__name__, template_folder=template_path)

# Load the model
model_path = os.path.join(current_dir, 'iris_model.pkl')
with open(model_path, 'rb') as f:
    model = pickle.load(f)

class_dict = {
    "0": "Iris setosa",
    "1": "Iris versicolor",
    "2": "Iris virginica"
}

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_class = None
    if request.method == 'POST':
        try:
            val1 = float(request.form['sepal_length'])
            val2 = float(request.form['sepal_width'])
            val3 = float(request.form['petal_length'])
            val4 = float(request.form['petal_width'])
            
            data = [[val1, val2, val3, val4]]
            prediction = str(model.predict(data)[0])
            pred_class = class_dict[prediction]
        except Exception as e:
            # Log error for debugging
            app.logger.error(f"Prediction error: {str(e)}")
            pred_class = "Error in prediction"
    
    return render_template('index.html', prediction=pred_class)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)