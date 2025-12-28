"""
Visualization Module for Dating Recommendation System
Creates publication-quality visualizations for data analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class DataVisualizer:
    """
    Visualization suite for dating profile data analysis
    Creates 5+ publication-quality visualizations
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.fig_counter = 0
        
    def plot_age_distribution(self, 
                               save_path: str = None,
                               figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Viz 3.1: Age distribution histogram with gender breakdown
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Age Distribution Analysis', fontsize=14, fontweight='bold')
        
        # Overall age distribution
        ax = axes[0]
        ax.hist(self.df['age'].dropna(), bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(self.df['age'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {self.df["age"].mean():.1f}')
        ax.axvline(self.df['age'].median(), color='green', linestyle='--', linewidth=2, label=f'Median: {self.df["age"].median():.1f}')
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Overall Age Distribution', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Age by gender
        ax = axes[1]
        if 'sex' in self.df.columns:
            for sex in self.df['sex'].unique():
                subset = self.df[self.df['sex'] == sex]['age'].dropna()
                label = 'Male' if sex == 'm' else 'Female' if sex == 'f' else str(sex)
                ax.hist(subset, bins=25, alpha=0.5, label=f'{label} (n={len(subset)})', edgecolor='black')
        ax.set_xlabel('Age', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Age Distribution by Gender', fontsize=12)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_category_frequency(self,
                                 save_path: str = None,
                                 figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """
        Viz 3.2: Category frequency analysis for key categorical variables
        """
        categorical_cols = ['orientation', 'status', 'body_type', 'diet', 'drinks', 'smokes']
        available_cols = [c for c in categorical_cols if c in self.df.columns][:6]
        
        n_plots = len(available_cols)
        n_rows = (n_plots + 1) // 2
        
        fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
        fig.suptitle('Category Frequency Analysis', fontsize=14, fontweight='bold')
        axes = axes.flatten()
        
        colors = sns.color_palette("viridis", 10)
        
        for i, col in enumerate(available_cols):
            ax = axes[i]
            value_counts = self.df[col].value_counts().head(8)
            
            bars = ax.barh(range(len(value_counts)), value_counts.values, color=colors)
            ax.set_yticks(range(len(value_counts)))
            ax.set_yticklabels(value_counts.index)
            ax.set_xlabel('Count', fontsize=10)
            ax.set_title(f'{col.replace("_", " ").title()} Distribution', fontsize=11)
            ax.invert_yaxis()
            
            # Add value labels
            for j, (bar, val) in enumerate(zip(bars, value_counts.values)):
                ax.text(val + 0.01 * max(value_counts), j, f'{val}', 
                       va='center', fontsize=9)
        
        # Hide unused subplots
        for i in range(len(available_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_top_locations(self,
                            top_n: int = 20,
                            save_path: str = None,
                            figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Viz 3.3: Top locations by user count
        """
        if 'location' not in self.df.columns:
            print("⚠️ Location column not found")
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get top locations
        location_counts = self.df['location'].value_counts().head(top_n)
        
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(location_counts)))
        
        bars = ax.barh(range(len(location_counts)), location_counts.values, color=colors)
        ax.set_yticks(range(len(location_counts)))
        ax.set_yticklabels([loc[:40] + '...' if len(str(loc)) > 40 else loc 
                           for loc in location_counts.index])
        ax.set_xlabel('Number of Users', fontsize=12)
        ax.set_title(f'Top {top_n} Locations by User Count', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for bar, val in zip(bars, location_counts.values):
            ax.text(val + 0.01 * max(location_counts), bar.get_y() + bar.get_height()/2,
                   f'{val:,}', va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self,
                                  save_path: str = None,
                                  figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Viz 3.4: Correlation heatmap of numerical features
        """
        # Select numerical columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ID columns
        numeric_cols = [c for c in numeric_cols if 'id' not in c.lower() and 'index' not in c.lower()]
        
        if len(numeric_cols) < 2:
            print("⚠️ Not enough numerical columns for correlation heatmap")
            return None
        
        # Calculate correlation
        corr_matrix = self.df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        sns.heatmap(corr_matrix, 
                    mask=mask,
                    annot=True, 
                    fmt='.2f', 
                    cmap='RdBu_r',
                    center=0,
                    square=True,
                    linewidths=0.5,
                    ax=ax,
                    cbar_kws={'shrink': 0.8})
        
        ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_interest_analysis(self,
                                save_path: str = None,
                                figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Viz 3.5: Interest distribution analysis
        """
        # Get interest columns
        interest_cols = [c for c in self.df.columns if c.startswith('interest_')]
        
        if not interest_cols:
            print("⚠️ Interest columns not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Interest Analysis', fontsize=14, fontweight='bold')
        
        # Interest frequency
        ax = axes[0]
        interest_counts = self.df[interest_cols].sum().sort_values(ascending=True)
        interest_names = [c.replace('interest_', '').title() for c in interest_counts.index]
        
        colors = plt.cm.Spectral(np.linspace(0.1, 0.9, len(interest_counts)))
        
        bars = ax.barh(interest_names, interest_counts.values, color=colors)
        ax.set_xlabel('Number of Users', fontsize=11)
        ax.set_title('Interest Popularity', fontsize=12)
        
        for bar, val in zip(bars, interest_counts.values):
            ax.text(val + 0.01 * max(interest_counts), bar.get_y() + bar.get_height()/2,
                   f'{int(val):,}', va='center', fontsize=9)
        
        # Interest co-occurrence
        ax = axes[1]
        if len(interest_cols) > 1:
            interest_corr = self.df[interest_cols].corr()
            interest_names_short = [c.replace('interest_', '')[:8] for c in interest_corr.columns]
            
            sns.heatmap(interest_corr, 
                       annot=True, 
                       fmt='.2f',
                       cmap='YlOrRd',
                       xticklabels=interest_names_short,
                       yticklabels=interest_names_short,
                       ax=ax,
                       cbar_kws={'shrink': 0.8})
            ax.set_title('Interest Co-occurrence', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_gender_orientation_distribution(self,
                                              save_path: str = None,
                                              figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Viz 3.6: Gender and orientation distribution
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Gender & Orientation Distribution', fontsize=14, fontweight='bold')
        
        # Gender distribution (pie chart)
        ax = axes[0]
        if 'sex' in self.df.columns:
            gender_counts = self.df['sex'].value_counts()
            labels = ['Male' if g == 'm' else 'Female' if g == 'f' else str(g) 
                     for g in gender_counts.index]
            colors = ['#4C72B0', '#DD8452', '#55A868'][:len(gender_counts)]
            
            wedges, texts, autotexts = ax.pie(gender_counts.values, 
                                              labels=labels,
                                              autopct='%1.1f%%',
                                              colors=colors,
                                              explode=[0.02] * len(gender_counts),
                                              shadow=True)
            ax.set_title('Gender Distribution', fontsize=12)
        
        # Orientation distribution (bar chart)
        ax = axes[1]
        if 'orientation' in self.df.columns:
            orientation_counts = self.df['orientation'].value_counts()
            colors = plt.cm.Set2(np.linspace(0, 1, len(orientation_counts)))
            
            ax.bar(range(len(orientation_counts)), orientation_counts.values, color=colors)
            ax.set_xticks(range(len(orientation_counts)))
            ax.set_xticklabels(orientation_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count', fontsize=11)
            ax.set_title('Sexual Orientation Distribution', fontsize=12)
            
            # Add value labels
            for i, val in enumerate(orientation_counts.values):
                ax.text(i, val + 0.02 * max(orientation_counts), f'{val:,}',
                       ha='center', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def plot_bio_length_analysis(self,
                                  save_path: str = None,
                                  figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """
        Viz 3.7: Bio text length analysis
        """
        if 'bio' not in self.df.columns:
            print("⚠️ Bio column not found")
            return None
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('Bio Text Analysis', fontsize=14, fontweight='bold')
        
        # Bio length distribution
        ax = axes[0]
        bio_lengths = self.df['bio'].fillna('').apply(len)
        
        ax.hist(bio_lengths, bins=50, edgecolor='black', alpha=0.7, color='teal')
        ax.axvline(bio_lengths.mean(), color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {bio_lengths.mean():.0f} chars')
        ax.set_xlabel('Bio Length (characters)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Bio Length Distribution', fontsize=12)
        ax.legend()
        ax.set_xlim(0, bio_lengths.quantile(0.95))
        
        # Bio length vs age scatter
        ax = axes[1]
        if 'age' in self.df.columns:
            sample_idx = np.random.choice(len(self.df), min(2000, len(self.df)), replace=False)
            sample_ages = self.df.iloc[sample_idx]['age'].values
            sample_lengths = bio_lengths.iloc[sample_idx].values
            
            ax.scatter(sample_ages, sample_lengths, alpha=0.3, s=10, c='purple')
            
            # Add trend line
            z = np.polyfit(sample_ages, sample_lengths, 1)
            p = np.poly1d(z)
            age_range = np.linspace(sample_ages.min(), sample_ages.max(), 100)
            ax.plot(age_range, p(age_range), 'r--', linewidth=2, label='Trend')
            
            ax.set_xlabel('Age', fontsize=11)
            ax.set_ylabel('Bio Length (characters)', fontsize=11)
            ax.set_title('Bio Length vs Age', fontsize=12)
            ax.legend()
            ax.set_ylim(0, bio_lengths.quantile(0.95))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Saved: {save_path}")
        
        return fig
    
    def generate_all_visualizations(self, 
                                     output_dir: str = './visualizations',
                                     show_plots: bool = False) -> Dict[str, plt.Figure]:
        """
        Generate all visualizations and save to directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n" + "="*60)
        print("📊 GENERATING VISUALIZATIONS")
        print("="*60)
        
        figures = {}
        
        # Generate each visualization
        viz_funcs = [
            ('age_distribution', self.plot_age_distribution),
            ('category_frequency', self.plot_category_frequency),
            ('top_locations', self.plot_top_locations),
            ('correlation_heatmap', self.plot_correlation_heatmap),
            ('interest_analysis', self.plot_interest_analysis),
            ('gender_orientation', self.plot_gender_orientation_distribution),
            ('bio_analysis', self.plot_bio_length_analysis),
        ]
        
        for name, func in viz_funcs:
            try:
                save_path = f"{output_dir}/{name}.png"
                fig = func(save_path=save_path)
                if fig:
                    figures[name] = fig
                    if not show_plots:
                        plt.close(fig)
            except Exception as e:
                print(f"⚠️ Error generating {name}: {e}")
        
        print(f"\n✅ Generated {len(figures)} visualizations in {output_dir}")
        
        return figures


def create_evaluation_plots(evaluation_results: Dict,
                            save_path: str = None,
                            figsize: Tuple[int, int] = (16, 12)) -> plt.Figure:
    """
    Create visualization plots for model evaluation results
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('Model Evaluation Results\n(Content-Based vs Hybrid)', 
                 fontsize=14, fontweight='bold')
    
    models = list(evaluation_results.keys())
    x = np.arange(len(models))
    width = 0.25
    
    # Define colors
    colors = {'5': '#4C72B0', '10': '#DD8452', '20': '#55A868'}
    
    # Precision@K
    ax = axes[0, 0]
    for i, k in enumerate([5, 10, 20]):
        values = [evaluation_results[m].get(f'precision@{k}', 0) for m in models]
        ax.bar(x + i*width, values, width, label=f'@{k}', color=colors[str(k)])
    ax.set_ylabel('Precision')
    ax.set_title('Precision@K by Model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Recall@K
    ax = axes[0, 1]
    for i, k in enumerate([5, 10, 20]):
        values = [evaluation_results[m].get(f'recall@{k}', 0) for m in models]
        ax.bar(x + i*width, values, width, label=f'@{k}', color=colors[str(k)])
    ax.set_ylabel('Recall')
    ax.set_title('Recall@K by Model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # NDCG@K
    ax = axes[0, 2]
    for i, k in enumerate([5, 10, 20]):
        values = [evaluation_results[m].get(f'ndcg@{k}', 0) for m in models]
        ax.bar(x + i*width, values, width, label=f'@{k}', color=colors[str(k)])
    ax.set_ylabel('NDCG')
    ax.set_title('NDCG@K by Model')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Other metrics (MRR, Coverage, Diversity)
    ax = axes[1, 0]
    metrics = ['mrr', 'coverage', 'diversity']
    metric_colors = ['#C44E52', '#8172B3', '#CCB974']
    
    for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
        values = [evaluation_results[m].get(metric, 0) for m in models]
        ax.bar(x + i*width, values, width, label=metric.upper(), color=color)
    ax.set_ylabel('Score')
    ax.set_title('Additional Metrics')
    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # RMSE (lower is better)
    ax = axes[1, 1]
    rmse_values = [evaluation_results[m].get('rmse', 0) for m in models]
    # Handle NaN values
    rmse_values = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in rmse_values]
    bars = ax.bar(x, rmse_values, width=0.6, color='#E74C3C', edgecolor='black')
    ax.set_ylabel('RMSE (lower is better)')
    ax.set_title('Root Mean Squared Error')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, rmse_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # MAE (lower is better)
    ax = axes[1, 2]
    mae_values = [evaluation_results[m].get('mae', 0) for m in models]
    # Handle NaN values
    mae_values = [v if not (isinstance(v, float) and np.isnan(v)) else 0 for v in mae_values]
    bars = ax.bar(x, mae_values, width=0.6, color='#3498DB', edgecolor='black')
    ax.set_ylabel('MAE (lower is better)')
    ax.set_title('Mean Absolute Error')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.grid(axis='y', alpha=0.3)
    # Add value labels on bars
    for bar, val in zip(bars, mae_values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 Saved: {save_path}")
    
    return fig
