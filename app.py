from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('iris_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from form
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
        iris_classes = ['Setosa', 'Versicolor', 'Virginica']
        result = iris_classes[prediction]
        return render_template('index.html', prediction_text=f'The predicted flower is: {result}')
    except:
        return render_template('index.html', prediction_text='Please enter valid numbers.')

if __name__ == '__main__':
    app.run(debug=True)
