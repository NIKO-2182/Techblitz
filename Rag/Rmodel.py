from flask import Flask, request, jsonify, Blueprint
from retrieve import generate_response, finance_keywords

app = Flask(__name__)

Rmodel = Blueprint("Finance", __name__)

@Rmodel.route("/Rmodel", methods=["POST"])
def chatbot_route():
  
    data = request.json
    prompt = data.get("prompt")
    result = generate_response(prompt)
    
    return jsonify("Answer :", result)

app.register_blueprint(Rmodel)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
