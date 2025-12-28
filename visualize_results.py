"""
Standalone script to visualize evaluation results from text file
Excludes Collaborative Filtering results
"""

import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def parse_evaluation_results(file_path):
    """Parse evaluation results from text file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = {}
    
    # Find each model section (skip Collaborative)
    model_sections = re.finditer(r'📊 (Content-Based|Hybrid)\n=+\n(.*?)(?=\n=|$)', content, re.DOTALL)
    
    for match in model_sections:
        model_name = match.group(1)
        metrics_text = match.group(2)
        
        # Parse metrics
        metrics = {}
        for line in metrics_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
        
        results[model_name] = metrics
    
    return results


def create_visualization(results, save_path='output/evaluation_results.png'):
    """Create visualization plots for evaluation results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle('Model Evaluation Results\n(Content-Based vs Hybrid)', 
                 fontsize=14, fontweight='bold')
    
    models = list(results.keys())
    x = np.arange(len(models))
    width = 0.25
    
    # Define colors
    colors = {'5': '#4C72B0', '10': '#DD8452', '20': '#55A868'}
    
    # 1. Precision@K
    ax = axes[0, 0]
    for i, k in enumerate(['5', '10', '20']):
        values = [results[m].get(f'precision@{k}', 0) for m in models]
        ax.bar(x + (i - 1) * width, values, width, label=f'K={k}', color=colors[k])
    ax.set_ylabel('Precision@K', fontsize=11)
    ax.set_title('Precision@K', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 2. Recall@K
    ax = axes[0, 1]
    for i, k in enumerate(['5', '10', '20']):
        values = [results[m].get(f'recall@{k}', 0) for m in models]
        ax.bar(x + (i - 1) * width, values, width, label=f'K={k}', color=colors[k])
    ax.set_ylabel('Recall@K', fontsize=11)
    ax.set_title('Recall@K', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 3. NDCG@K
    ax = axes[0, 2]
    for i, k in enumerate(['5', '10', '20']):
        values = [results[m].get(f'ndcg@{k}', 0) for m in models]
        ax.bar(x + (i - 1) * width, values, width, label=f'K={k}', color=colors[k])
    ax.set_ylabel('NDCG@K', fontsize=11)
    ax.set_title('NDCG@K (Ranking Quality)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 4. MRR, Coverage, Diversity
    ax = axes[1, 0]
    metric_names = ['mrr', 'coverage', 'diversity']
    metric_labels = ['MRR', 'Coverage', 'Diversity']
    width_multi = 0.2
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        values = [results[m].get(metric, 0) for m in models]
        ax.bar(x + (i - 1) * width_multi, values, width_multi, label=label)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('MRR / Coverage / Diversity', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 5. RMSE
    ax = axes[1, 1]
    values = [results[m].get('rmse', 0) for m in models]
    bars = ax.bar(x, values, color='coral', alpha=0.7, edgecolor='black')
    ax.set_ylabel('RMSE', fontsize=11)
    ax.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 6. MAE
    ax = axes[1, 2]
    values = [results[m].get('mae', 0) for m in models]
    bars = ax.bar(x, values, color='lightblue', alpha=0.7, edgecolor='black')
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('MAE (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Visualization saved to: {save_path}")
    
    return fig


if __name__ == '__main__':
    # Parse results
    print("📖 Parsing evaluation results...")
    results = parse_evaluation_results('output/evaluation_results.txt')
    
    print(f"✅ Found {len(results)} models: {', '.join(results.keys())}")
    
    # Create visualization
    print("\n🎨 Creating visualization...")
    create_visualization(results, save_path='output/evaluation_results.png')
    
    print("\n✅ Done!")
