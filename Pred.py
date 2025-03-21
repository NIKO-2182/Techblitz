from flask import Flask, request, jsonify, Blueprint
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from Techblitz.Visual.previsual import load_and_prepare_data
from Techblitz.Visual.Arima import train_all_metrics
from Techblitz.Visual.Postvisual import (
    adjust_predictions, 
    format_predictions, 
    visualize_predictions
)

# Initialize Flask app and Blueprint
app = Flask(__name__)
analysis = Blueprint("analysis", __name__, url_prefix="/api")

@analysis.route("/analyze", methods=["POST"])
def analyze_financial_data():
    """API endpoint for financial analysis"""
    try:
        # Validate request data
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'status': 400
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 400
            }), 400

        # Get user parameters
        try:
            user_params = pd.DataFrame([{
                'Inflation_Rate': float(request.form.get('inflation_rate', 0)),
                'Interest_Rate': float(request.form.get('interest_rate', 0)),
                'Growth_Factor': float(request.form.get('growth_factor', 1))
            }])
        except ValueError:
            return jsonify({
                'error': 'Invalid parameter values',
                'status': 400
            }), 400

        # Read and process file
        file_data = file.read()
        data, viz_files = load_and_prepare_data(file_data)
        
        if data is None:
            return jsonify({
                'error': 'Failed to load data',
                'status': 500
            }), 500

        # Define metrics
        targets = ['Revenue_Growth', 'Profit_Margin', 'Cash_Flow']
        
        # Train models and get predictions
        results_dict = train_all_metrics(data, targets)
        predictions_summary = []

        # Process predictions for each metric
        for target in targets:
            original_pred = results_dict[target]['predictions'][-1]
            adjusted_pred = adjust_predictions(
                {target: original_pred}, 
                user_params
            )[target]
            
            prediction_info = format_predictions(
                original_pred, 
                adjusted_pred, 
                target
            )
            predictions_summary.append(prediction_info)

        # Generate prediction visualizations
        pred_viz_files = visualize_predictions(predictions_summary)

        # Save predictions to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        filename = f'{results_dir}/predictions_{timestamp}.csv'
        pd.DataFrame(predictions_summary).to_csv(filename, index=False)

        return jsonify({
            'status': 200,
            'predictions': predictions_summary,
            'visualizations': {
                'data': viz_files,
                'predictions': pred_viz_files
            },
            'file_saved': filename
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 500
        }), 500

# Register blueprint
app.register_blueprint(analysis)

if __name__ == "__main__":
    print("\n🚀 Starting Financial Analysis API...")
    print("=" * 50)
    print("📍 Endpoint: http://localhost:5003/api/analyze")
    print("📁 Accepts: CSV/PDF file upload")
    print("📊 Parameters:")
    print("  • inflation_rate (float)")
    print("  • interest_rate (float)")
    print("  • growth_factor (float)")
    print("\n💾 Outputs:")
    print("  • Prediction results (JSON)")
    print("  • Data visualizations (PNG)")
    print("  • Prediction visualizations (PNG)")
    print("  • Results summary (CSV)")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5003, debug=True)