from flask import Flask, jsonify,request
import tensorflow as tf
app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict(): 
    bedrooms = request.form['bedrooms']
    bathrooms = request.form['bathrooms']
    totalsqft = request.form['totalsqft']
    floors = request.form['floors']
    sqft_above = request.form['sqft_above']
    sqft_basement = request.form['sqft_basement']
    pre=""
    model = tf.keras.models.load_model("price_model.h5")

    pred = model.predict([[int(bedrooms), int(bathrooms), int(totalsqft), int(floors), int(sqft_above),int(sqft_basement)]])
    p=pred.tolist()
    for ele in p[0]:
        pre += str(ele)
    return jsonify(price= pre)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
