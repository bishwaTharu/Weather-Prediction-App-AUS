import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_eda(filepath, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Loading data from {filepath}...")
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found.")
        return
        
    df = pd.read_csv(filepath)
    
    logger.info("Computing summary statistics...")
    # Summary statistics: mean, median, std, etc.
    summary_stats = df.describe().T
    summary_stats['median'] = df.median(numeric_only=True)
    
    # Reorder columns to highlight mean, median, std
    cols = ['mean', 'median', 'std', 'min', '25%', '50%', '75%', 'max', 'count']
    summary_stats = summary_stats[[c for c in cols if c in summary_stats.columns]]
    
    stats_file = os.path.join(output_dir, "summary_statistics.csv")
    summary_stats.to_csv(stats_file)
    logger.info(f"Summary statistics saved to {stats_file}")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    logger.info("Generating Visualizations...")
    
    # 1. Histogram (e.g., MaxTemp)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['MaxTemp'].dropna(), bins=30, kde=True, color='skyblue')
    plt.title('Distribution of Maximum Temperature (MaxTemp)')
    plt.xlabel('MaxTemp')
    plt.ylabel('Frequency')
    plt.tight_layout()
    hist_path = os.path.join(output_dir, 'histogram_maxtemp.png')
    plt.savefig(hist_path)
    plt.close()
    logger.info(f"Saved histogram to {hist_path}")
    
    # 2. Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Numerical Features')
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, 'correlation_heatmap.png')
    plt.savefig(heatmap_path)
    plt.close()
    logger.info(f"Saved correlation heatmap to {heatmap_path}")
    
    # 3. Scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='MinTemp', y='MaxTemp', data=df, alpha=0.5, hue='RainToday')
    plt.title('Scatter plot: MinTemp vs MaxTemp')
    plt.xlabel('Minimum Temperature')
    plt.ylabel('Maximum Temperature')
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, 'scatter_mintemp_maxtemp.png')
    plt.savefig(scatter_path)
    plt.close()
    logger.info(f"Saved scatter plot to {scatter_path}")
    
    # 4. Generate some text-based insights
    logger.info("Generating key insights...")
    insights = []
    
    # Missing values insight
    total_missing = df.isnull().sum().sum()
    insights.append(f"The dataset contains a total of {total_missing} missing values.")
    
    # Correlation insight
    # Find highly correlated pairs
    corr_unstacked = corr_matrix.abs().unstack()
    high_corr = corr_unstacked[(corr_unstacked > 0.8) & (corr_unstacked < 1.0)].drop_duplicates()
    if not high_corr.empty:
        insights.append("Highly correlated features include:")
        for idx, val in high_corr.items():
            insights.append(f"  - {idx[0]} and {idx[1]} (Correlation: {corr_matrix.loc[idx[0], idx[1]]:.2f})")
    
    insights_path = os.path.join(output_dir, 'key_insights.txt')
    with open(insights_path, 'w') as f:
        f.write("KEY INSIGHTS FROM DATA\n")
        f.write("======================\n\n")
        for insight in insights:
            f.write(insight + "\n")
            
    logger.info(f"Saved key insights to {insights_path}")
    logger.info("EDA completed successfully!")

if __name__ == "__main__":
    train_file = "data/Weather Training Data.csv"
    run_eda(train_file, "assets")
