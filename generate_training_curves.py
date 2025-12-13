#!/usr/bin/env python3
"""
Generate training curves figure matching the reference style.
Exports validation loss (MSE) over epochs for Shortlist, Longlist, and Original (Full) feature sets.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from pathlib import Path

# Configure matplotlib for publication-quality figures
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['axes.titlesize'] = 16
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['axes.linewidth'] = 1.2
mpl.rcParams['xtick.major.width'] = 1.2
mpl.rcParams['ytick.major.width'] = 1.2

def load_training_metrics():
    """Load training metrics for all feature sets."""
    base_path = Path('data_eval')
    
    # Load baseline training metrics for each feature set
    shortlist = pd.read_csv(base_path / 'nn_training_metrics_shortlist_baseline.csv')
    longlist = pd.read_csv(base_path / 'nn_training_metrics_longlist_baseline.csv')
    full = pd.read_csv(base_path / 'nn_training_metrics_full_baseline.csv')
    
    return shortlist, longlist, full

def get_final_test_r2(df):
    """Get the final verification (test) R² from the training metrics."""
    # verif_r2 is the R² on a held-out verification set (test set)
    return df['verif_r2'].iloc[-1]

def create_training_curves_figure(shortlist, longlist, full, output_path, format='pdf', max_epochs=None):
    """Create training curves figure matching reference style.
    
    Args:
        shortlist, longlist, full: DataFrames with training metrics
        output_path: Path to save figure
        format: Output format ('pdf' or 'png')
        max_epochs: If set, truncate data to this many epochs (to match reference)
    """
    
    # Optionally truncate to match reference image (which shows 60 epochs)
    if max_epochs is not None:
        shortlist = shortlist[shortlist['epoch'] <= max_epochs]
        longlist = longlist[longlist['epoch'] <= max_epochs]
        full = full[full['epoch'] <= max_epochs]
    
    # Get final R² values for legend (use verif_r2 which is the test set R²)
    shortlist_r2 = get_final_test_r2(shortlist)
    longlist_r2 = get_final_test_r2(longlist)
    full_r2 = get_final_test_r2(full)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot validation MSE curves (matching the reference image style)
    # Use the standard matplotlib color cycle (tab10)
    colors = plt.cm.tab10.colors
    
    ax.plot(shortlist['epoch'], shortlist['val_mse'], 
            color=colors[0], linewidth=2, 
            label=f'Shortlist (R2: {shortlist_r2:.3f})')
    
    ax.plot(longlist['epoch'], longlist['val_mse'], 
            color=colors[1], linewidth=2, 
            label=f'Longlist (R2: {longlist_r2:.3f})')
    
    ax.plot(full['epoch'], full['val_mse'], 
            color=colors[2], linewidth=2, 
            label=f'Original (R2: {full_r2:.3f})')
    
    # Style the plot to match reference
    ax.set_xlabel('Epoch', fontsize=14, fontweight='normal')
    ax.set_ylabel('Mean Squared Error', fontsize=14, fontweight='normal')
    ax.set_title('Validation Loss (MSE) over Epochs', fontsize=16, fontweight='normal')
    
    # Set axis limits
    if max_epochs is not None:
        ax.set_xlim(0, max_epochs)
    else:
        ax.set_xlim(0, 100)
    
    # Add legend in upper right corner with a box
    ax.legend(loc='upper right', frameon=True, framealpha=1.0, 
              edgecolor='black', fancybox=False)
    
    # Add grid for readability (light gray, behind the data)
    ax.grid(True, linestyle='-', alpha=0.3, color='gray')
    ax.set_axisbelow(True)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_file = Path(output_path)
    plt.savefig(output_file, format=format, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved figure to {output_file}")
    
    # Also save as PNG for preview
    if format == 'pdf':
        png_path = output_file.with_suffix('.png')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Also saved PNG preview to {png_path}")
    
    plt.close()

def export_combined_training_log(shortlist, longlist, full, output_path):
    """Export combined training log CSV with epoch, train/val loss and R² for all feature sets."""
    
    combined = pd.DataFrame({
        'epoch': shortlist['epoch'],
        # Shortlist
        'shortlist_train_loss': shortlist['train_loss'],
        'shortlist_train_mse': shortlist['train_mse'],
        'shortlist_train_r2': shortlist['train_r2'],
        'shortlist_val_loss': shortlist['val_loss'],
        'shortlist_val_mse': shortlist['val_mse'],
        'shortlist_val_r2': shortlist['val_r2'],
        # Longlist
        'longlist_train_loss': longlist['train_loss'],
        'longlist_train_mse': longlist['train_mse'],
        'longlist_train_r2': longlist['train_r2'],
        'longlist_val_loss': longlist['val_loss'],
        'longlist_val_mse': longlist['val_mse'],
        'longlist_val_r2': longlist['val_r2'],
        # Full (Original)
        'full_train_loss': full['train_loss'],
        'full_train_mse': full['train_mse'],
        'full_train_r2': full['train_r2'],
        'full_val_loss': full['val_loss'],
        'full_val_mse': full['val_mse'],
        'full_val_r2': full['val_r2'],
    })
    
    combined.to_csv(output_path, index=False)
    print(f"Saved combined training log to {output_path}")
    
    return combined

def main():
    print("Loading training metrics...")
    shortlist, longlist, full = load_training_metrics()
    
    print(f"\nDataset shapes:")
    print(f"  Shortlist: {len(shortlist)} epochs")
    print(f"  Longlist: {len(longlist)} epochs")
    print(f"  Full: {len(full)} epochs")
    
    # Print final R² values
    print(f"\nFinal Test R² values:")
    print(f"  Shortlist: {get_final_test_r2(shortlist):.4f}")
    print(f"  Longlist: {get_final_test_r2(longlist):.4f}")
    print(f"  Full (Original): {get_final_test_r2(full):.4f}")
    
    # Create output directory
    output_dir = Path('figures')
    output_dir.mkdir(exist_ok=True)
    
    # Export combined training log
    print("\nExporting combined training log...")
    export_combined_training_log(
        shortlist, longlist, full,
        'data_eval/nn_combined_training_log.csv'
    )
    
    # Create PDF figure (full 100 epochs)
    print("\nGenerating training curves figure (100 epochs)...")
    create_training_curves_figure(
        shortlist, longlist, full,
        'figures/nn_validation_loss_curves.pdf',
        format='pdf'
    )
    
    # Also create a version matching reference image (60 epochs)
    print("\nGenerating training curves figure (60 epochs - matching reference)...")
    create_training_curves_figure(
        shortlist, longlist, full,
        'figures/nn_validation_loss_curves_60ep.pdf',
        format='pdf',
        max_epochs=60
    )
    
    print("\nDone!")

if __name__ == '__main__':
    main()

