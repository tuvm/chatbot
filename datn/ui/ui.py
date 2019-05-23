import os.path, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))
from flask import Flask, render_template, request, json
import answer

app = Flask(__name__)

answer_obj = answer.Answer()

@app.route('/')
def index():
    return render_template('ui.html')

@app.route('/get_reply')
def get_reply():
    message = request.args.get('message')
    if message:
        data = answer_obj.answer(message)
        response = app.response_class(
            response=json.dumps(data),
            status=200,
            mimetype='application/json')
    else:
        data = 'Invalid message'
        response = app.response_class(
            response=json.dumps(data),
            status=404,
            mimetype='application/json')

    return response


if __name__ == '__main__':
    app.run(debug=True)
