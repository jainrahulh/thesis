from flask import Flask, render_template, request
from joblib import load


pipeline = load("news_classifier.joblib")


def requestResults(text):
    result = (pipeline.predict([text]))
    return result


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET'])
def get_data():
    import sys
    if request.method == 'POST':
        return render_template('index.html')

@app.route('/predict', methods=['POST'])
def classify():
    inputText = requestResults(request.form['classify'])
    print(inputText)
    return render_template('index.html', prediction_text=str(inputText))


if __name__ == '__main__' :
    app.run(debug=True)