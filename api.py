import flask
import model

app = flask.Flask(__name__)
app.config["DEBUG"] = True

model = model.Model()
@app.route('/predict/<path:input_url>', methods=['GET'])
def api(input_url):
    return model.predict(input_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8081)