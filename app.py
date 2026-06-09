from flask import Flask, request, jsonify
from flask_cors import CORS
from main import predict_strategy
from main import train_model

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json()
    driver = data['driver']
    circuit = data['circuit']

    reg, feature_columns, multi_year_laps, session, avg_delta_soft, avg_delta_medium, avg_delta_hard = train_model(driver, circuit, [2018, 2019, 2020, 2021, 2022, 2023, 2024])
    best_strategy, best_compound1, best_compound2, best_time = predict_strategy(reg, feature_columns, multi_year_laps, session, avg_delta_soft, avg_delta_medium, avg_delta_hard)
    
    return jsonify({
        "best_strategy": best_strategy,
        "best_compound1": best_compound1,
        "best_compound2": best_compound2,
        "best_time": best_time
    })

if __name__ == "__main__":
    app.run(debug=True)