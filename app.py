from flask import Flask, request, render_template
import joblib
import numpy as np


app = Flask(__name__)

model = joblib.load('knn_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['sepal_length']),
            float(request.form['sepal_width']),
            float(request.form['petal_length']),
            float(request.form['petal_width'])
        ]

        input_arry = np.array([features])
        prediction = model.predict(input_arry)[0]
        
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        flower_name = iris_classes[prediction]
        return render_template('result.html', prediction=flower_name)

    except ValueError:
        return "❌ Invalid input. Please enter numeric values for all features."
    
if __name__ == '__main__':
    app.run(debug=True)