from flask import Flask,  render_template,request
import numpy as np
import pickle
model1 = pickle.load(open(r"model1.pk1", 'rb'))
app = Flask(__name__, template_folder='template')
# render html page
@app.route("/Home")
def home():
    return render_template('Home.html')
@app.route("/predict")
def home1():
    return render_template('predict.html')
@app.route("/pred", methods=['POST', 'GET'])
def predict():
    x = [[int(x) for x in request.form.values()]]
    print(x)

    x = np.array(x)
    print(x.shape)

    print(x)
    pred = model1.predict(x)
    print(pred)
    return render_template('submit.html', prediction_text=pred)

# main function
if __name__ == "__main__":
    app.run(debug=False)


