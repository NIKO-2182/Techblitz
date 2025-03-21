import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

def visualize_predictions(predictions_data, viz_dir='visualizations'):
    """Generate and save visualization of predictions"""
    try:
        # Create visualization directory if it doesn't exist
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Comparison Bar Plot
        plt.figure(figsize=(12, 6))
        metrics = [p['metric'] for p in predictions_data]
        orig_vals = [p['original'] for p in predictions_data]
        adj_vals = [p['adjusted'] for p in predictions_data]
        
        x = range(len(metrics))
        width = 0.35
        
        plt.bar([i - width/2 for i in x], orig_vals, width, label='Original', color='skyblue')
        plt.bar([i + width/2 for i in x], adj_vals, width, label='Adjusted', color='lightcoral')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Original vs Adjusted Predictions')
        plt.xticks(x, metrics, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        comparison_file = f'{viz_dir}/prediction_comparison_{timestamp}.png'
        plt.savefig(comparison_file)
        plt.close()
        
        # 2. Change Percentage Plot
        plt.figure(figsize=(10, 6))
        changes = [p['change_percent'] for p in predictions_data]
        
        colors = ['green' if c >= 0 else 'red' for c in changes]
        plt.bar(metrics, changes, color=colors)
        
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.2)
        plt.xlabel('Metrics')
        plt.ylabel('Change (%)')
        plt.title('Prediction Changes After Adjustment')
        plt.xticks(rotation=45)
        
        for i, v in enumerate(changes):
            plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom' if v >= 0 else 'top')
        
        plt.tight_layout()
        
        changes_file = f'{viz_dir}/prediction_changes_{timestamp}.png'
        plt.savefig(changes_file)
        plt.close()
        
        return {
            'comparison': comparison_file,
            'changes': changes_file
        }
        
    except Exception as e:
        print(f"❌ Error generating visualizations: {str(e)}")
        return None
def adjust_predictions(predictions, user_params):
    """
    Adjust predictions based on user parameters
    
    Args:
        predictions (dict): Original predictions for each metric
        user_params (pd.DataFrame): User input parameters
    
    Returns:
        dict: Adjusted predictions
    """
    try:
        # Extract parameters
        inflation = user_params['Inflation_Rate'].values[0] / 100
        interest = user_params['Interest_Rate'].values[0] / 100
        growth = user_params['Growth_Factor'].values[0]

        # Initialize adjusted predictions
        adjusted = {}

        for metric, value in predictions.items():
            # Apply different adjustments based on metric type
            if metric == 'Revenue_Growth':
                # Revenue affected by growth and inflation
                adjustment = (1 + growth - inflation)
            elif metric == 'Profit_Margin':
                # Margin affected by both rates negatively
                adjustment = (1 - inflation - interest)
            elif metric == 'Cash_Flow':
                # Cash flow affected by all factors
                adjustment = (1 + growth - (inflation + interest)/2)
            else:
                # Default adjustment if metric unknown
                adjustment = 1.0

            # Apply adjustment and store
            adjusted[metric] = float(value) * adjustment

        return adjusted

    except Exception as e:
        print(f"❌ Error in adjust_predictions: {str(e)}")
        return None
    
def format_predictions(original_pred, adjusted_pred, target):
    """Format prediction results for display"""
    return {
        'metric': target,
        'original': float(original_pred),
        'adjusted': float(adjusted_pred),
        'change_percent': ((adjusted_pred - original_pred) / original_pred) * 100,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ...existing code for get_user_input() and adjust_predictions()...