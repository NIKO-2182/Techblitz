from flask import Flask, request, jsonify, Blueprint
from werkzeug.utils import secure_filename
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tabula.io import read_pdf
from Techblitz.Visual.Arima import train_all_metrics
from Techblitz.Visual.Postvisual import (
    adjust_predictions, 
    format_predictions, 
    visualize_predictions
)

# Initialize Flask app and Blueprint
app = Flask(__name__)
analysis = Blueprint("analysis", __name__, url_prefix="/api")

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_financial_data(file_path):
    """Process uploaded financial data file"""
    try:
        # Load data based on file type
        if file_path.lower().endswith('.pdf'):
            tables = read_pdf(file_path, pages='all')
            if not tables:
                raise ValueError("No tables found in PDF")
            data = tables[0]
        else:
            data = pd.read_csv(file_path)

        # Validate required columns
        required_columns = [
            'Date', 'Inflation_Rate', 'Interest_Rate',
            'Revenue_Growth', 'Profit_Margin', 'Cash_Flow'
        ]
        
        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {', '.join(missing_cols)}")

        # Process data
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        
        # Convert numeric columns
        numeric_cols = data.columns.difference(['Date'])
        data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        if data.isnull().any().any():
            raise ValueError("Dataset contains missing values")

        # Get user parameters
        try:
            user_params = pd.DataFrame([{
                'Inflation_Rate': float(request.form.get('inflation_rate', 0)),
                'Interest_Rate': float(request.form.get('interest_rate', 0)),
                'Growth_Factor': float(request.form.get('growth_factor', 1))
            }])
        except ValueError:
            raise ValueError("Invalid parameter values")

        # Validate parameters
        if not (0 <= user_params['Growth_Factor'].values[0] <= 2):
            raise ValueError("Growth Factor must be between 0 and 2")
        if user_params['Inflation_Rate'].values[0] < 0:
            raise ValueError("Inflation Rate cannot be negative")
        if user_params['Interest_Rate'].values[0] < 0:
            raise ValueError("Interest Rate cannot be negative")

        # Process predictions
        targets = ['Revenue_Growth', 'Profit_Margin', 'Cash_Flow']
        results_dict = train_all_metrics(data, targets)
        predictions_summary = []

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

        # Generate visualizations
        viz_files = visualize_predictions(predictions_summary)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        filename = f'{results_dir}/predictions_{timestamp}.csv'
        pd.DataFrame(predictions_summary).to_csv(filename, index=False)

        return jsonify({
            'status': 200,
            'predictions': predictions_summary,
            'visualizations': viz_files,
            'file_saved': filename
        })

    except Exception as e:
        raise ValueError(f"Data processing failed: {str(e)}")

@analysis.route("/analyze", methods=["POST"])
def analyze_financial_data():
    """API endpoint for financial analysis"""
    try:
        # Ensure upload directory exists
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])

        # Validate file upload
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part in request',
                'details': 'Include file in form data with key "file"',
                'status': 400
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'details': 'Select a file before uploading',
                'status': 400
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                'error': 'Invalid file type',
                'details': f'Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                'status': 400
            }), 400

        # Save and process file
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            return process_financial_data(file_path)
            
        except ValueError as e:
            return jsonify({
                'error': 'Processing failed',
                'details': str(e),
                'status': 400
            }), 400
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    except Exception as e:
        return jsonify({
            'error': 'Server error',
            'details': str(e),
            'status': 500
        }), 500

# Register blueprint
app.register_blueprint(analysis)

if __name__ == "__main__":
    print("\n🚀 Starting Financial Analysis API...")
    print("=" * 50)
    print("📍 Endpoint: http://localhost:5003/api/analyze")
    print("📁 Accepts: PDF/CSV upload (max 16MB)")
    print("📊 Parameters:")
    print("  • inflation_rate (float, >= 0)")
    print("  • interest_rate (float, >= 0)")
    print("  • growth_factor (float, 0-2)")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5003, debug=True)