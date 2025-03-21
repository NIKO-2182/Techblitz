from flask import request
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tabula
import os
from datetime import datetime

def load_and_prepare_data(file_data=None):
    """Load financial data from uploaded CSV or PDF file and save visualizations"""
    try:
        if file_data is None:
            raise ValueError("No file data provided")

        # Create file-like object from uploaded data
        file_obj = io.BytesIO(file_data)
        
        # Get file extension from content type
        content_type = request.content_type
        is_csv = 'csv' in content_type.lower()
        is_pdf = 'pdf' in content_type.lower()

        # Load data based on file type
        if is_csv:
            data = pd.read_csv(file_obj)
        elif is_pdf:
            tables = tabula.read_pdf(file_obj, pages='all')
            if not tables:
                raise ValueError("No tables found in PDF")
            data = tables[0]
        else:
            raise ValueError("Unsupported file format. Please use CSV or PDF")

        # Validate required columns
        required_columns = [
            'Date', 'Inflation_Rate', 'Interest_Rate',
            'Revenue_Growth', 'Profit_Margin', 'Cash_Flow'
        ]
        
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Missing required columns. Need: {', '.join(required_columns)}")

        # Convert Date column to datetime index
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.set_index('Date')
        data = data.sort_index()
        
        # Create visualization directory
        viz_dir = 'visualizations'
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Time Series Plot
        plt.figure(figsize=(15, 10))
        for column in data.columns:
            plt.plot(data.index, data[column], label=column, marker='o')
        plt.title('Financial Metrics Over Time', fontsize=14, pad=20)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/timeseries_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation Heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(data.corr(), dtype=bool))
        sns.heatmap(data.corr(), 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   fmt='.2f',
                   square=True,
                   linewidths=1)
        plt.title('Correlation Between Metrics', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/correlation_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Individual Metrics Analysis
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Individual Metric Analysis', fontsize=16, y=1.02)
        
        for idx, column in enumerate(data.columns):
            row = idx // 3
            col = idx % 3
            
            # Plot with both line and points
            sns.lineplot(data=data, x=data.index, y=column, ax=axes[row, col])
            axes[row, col].scatter(data.index, data[column], color='red', alpha=0.5)
            
            # Add trend line
            z = np.polyfit(range(len(data.index)), data[column], 1)
            p = np.poly1d(z)
            axes[row, col].plot(data.index, p(range(len(data.index))), 
                              "r--", alpha=0.8, label='Trend')
            
            axes[row, col].set_title(f'{column} Trend', fontsize=12)
            axes[row, col].tick_params(axis='x', rotation=45)
            axes[row, col].grid(True, alpha=0.3)
            axes[row, col].legend()
        
        plt.tight_layout()
        plt.savefig(f'{viz_dir}/metrics_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create summary statistics
        stats_summary = data.describe()
        stats_file = f'{viz_dir}/statistics_{timestamp}.csv'
        stats_summary.to_csv(stats_file)
        
        # Create visualization files dictionary
        visualization_files = {
            'timeseries': f'timeseries_{timestamp}.png',
            'correlation': f'correlation_{timestamp}.png',
            'metrics': f'metrics_{timestamp}.png',
            'statistics': f'statistics_{timestamp}.csv'
        }
        
        return data, visualization_files

    except Exception as e:
        print(f"❌ Error in data processing: {str(e)}")
        return None, None

def validate_data(data):
    """Validate the loaded data"""
    try:
        if data is None:
            return False, "No data provided"
            
        if data.empty:
            return False, "Empty dataset"
            
        if data.isnull().any().any():
            return False, "Dataset contains missing values"
            
        return True, "Data validation successful"
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"