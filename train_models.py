#!/usr/bin/env python3
"""
Training script for Superconductor Critical Temperature Prediction
Models: Random Forest, XGBoost, Neural Network
Dataset split: 70% train / 15% validation / 15% test
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pathlib import Path
import argparse


# ============================================================
# Dataset
# ============================================================
class SuperconductorDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ============================================================
# Neural Network Model
# ============================================================
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.2):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, h),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = h
        layers.append(nn.Linear(prev_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()


# ============================================================
# Data Loading & Split
# ============================================================
def load_data(data_path, feature_list=None):
    """Load data and perform 70/15/15 split."""
    df = pd.read_csv(data_path)
    target_col = 'critical_temp' if 'critical_temp' in df.columns else df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if feature_list is not None:
        feature_list = [f for f in feature_list if f in X.columns]
        X = X[feature_list]

    # Split: 70/15/15
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    val_fraction = 0.15 / 0.85  # ~17.65% of temp => 15% overall
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction, random_state=42
    )

    print(f"Dataset split (random_state=42):")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ============================================================
# Train Random Forest
# ============================================================
def train_rf(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n=== Training Random Forest ===")
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    val_r2 = r2_score(y_val, rf.predict(X_val))
    test_r2 = r2_score(y_test, rf.predict(X_test))
    print(f"  Val R²:  {val_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    return {'model': 'Random Forest', 'val_r2': val_r2, 'test_r2': test_r2}


# ============================================================
# Train XGBoost
# ============================================================
def train_xgb(X_train, X_val, X_test, y_train, y_val, y_test):
    print("\n=== Training XGBoost ===")
    xgb = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)

    val_r2 = r2_score(y_val, xgb.predict(X_val))
    test_r2 = r2_score(y_test, xgb.predict(X_test))
    print(f"  Val R²:  {val_r2:.4f}")
    print(f"  Test R²: {test_r2:.4f}")
    return {'model': 'XGBoost', 'val_r2': val_r2, 'test_r2': test_r2}


# ============================================================
# Train Neural Network
# ============================================================
def train_nn(X_train, X_val, X_test, y_train, y_val, y_test,
             epochs=120, batch_size=64, lr=0.001, weight_decay=1e-5,
             patience=15, save_metrics_path=None):
    print("\n=== Training Neural Network ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr_s = scaler_X.fit_transform(X_train)
    X_va_s = scaler_X.transform(X_val)
    X_te_s = scaler_X.transform(X_test)
    y_tr_s = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_va_s = scaler_y.transform(y_val.values.reshape(-1, 1)).flatten()

    train_ds = SuperconductorDataset(X_tr_s, y_tr_s)
    val_ds = SuperconductorDataset(X_va_s, y_va_s)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = SimpleNN(X_tr_s.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    best_val_loss = float('inf')
    patience_ctr = 0
    best_state = None
    metrics_history = []

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                loss = criterion(out, yb)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Compute R² in original scale
        model.eval()
        with torch.no_grad():
            train_pred_s = model(torch.FloatTensor(X_tr_s).to(device)).cpu().numpy()
            val_pred_s = model(torch.FloatTensor(X_va_s).to(device)).cpu().numpy()
        train_pred = scaler_y.inverse_transform(train_pred_s.reshape(-1, 1)).flatten()
        val_pred = scaler_y.inverse_transform(val_pred_s.reshape(-1, 1)).flatten()
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)

        metrics_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_r2': train_r2,
            'val_r2': val_r2,
        })

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_ctr = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                  f"train_R²={train_r2:.4f}, val_R²={val_r2:.4f}")

        if patience_ctr >= patience:
            print(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best
    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Final evaluation
    with torch.no_grad():
        val_pred_s = model(torch.FloatTensor(X_va_s).to(device)).cpu().numpy()
        test_pred_s = model(torch.FloatTensor(X_te_s).to(device)).cpu().numpy()
    val_pred = scaler_y.inverse_transform(val_pred_s.reshape(-1, 1)).flatten()
    test_pred = scaler_y.inverse_transform(test_pred_s.reshape(-1, 1)).flatten()
    final_val_r2 = r2_score(y_val, val_pred)
    final_test_r2 = r2_score(y_test, test_pred)

    print(f"  Best Val R²:  {final_val_r2:.4f}")
    print(f"  Test R²:      {final_test_r2:.4f}")

    # Save metrics CSV
    if save_metrics_path:
        pd.DataFrame(metrics_history).to_csv(save_metrics_path, index=False)
        print(f"  Saved training metrics to {save_metrics_path}")

    return {
        'model': 'Neural Network',
        'val_r2': final_val_r2,
        'test_r2': final_test_r2,
        'metrics_history': metrics_history,
    }


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Train models for superconductor Tc prediction')
    parser.add_argument('--data', default='train.csv', help='Path to training data CSV')
    parser.add_argument('--features', default='full', choices=['shortlist', 'longlist', 'full'],
                        help='Feature set to use')
    parser.add_argument('--output', default='data_eval', help='Output directory for metrics')
    args = parser.parse_args()

    root = Path('.').resolve()
    output_dir = root / args.output
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load feature lists
    feature_list = None
    if args.features == 'shortlist':
        feature_list = pd.read_csv('data_eval/important_features_shortlist_superconductivity.csv')['feature'].tolist()
    elif args.features == 'longlist':
        feature_list = pd.read_csv('data_eval/important_features_longlist_superconductivity.csv')['feature'].tolist()

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data, feature_list)
    print(f"\nFeature set: {args.features} ({X_train.shape[1]} features)")

    # Train all models
    results = []
    results.append(train_rf(X_train, X_val, X_test, y_train, y_val, y_test))
    results.append(train_xgb(X_train, X_val, X_test, y_train, y_val, y_test))

    nn_metrics_path = output_dir / f'nn_training_metrics_{args.features}.csv'
    nn_result = train_nn(X_train, X_val, X_test, y_train, y_val, y_test,
                         save_metrics_path=nn_metrics_path)
    results.append({'model': nn_result['model'], 'val_r2': nn_result['val_r2'], 'test_r2': nn_result['test_r2']})

    # Summary
    print("\n" + "=" * 50)
    print(f"RESULTS SUMMARY ({args.features} features)")
    print("=" * 50)
    for r in results:
        print(f"  {r['model']:15s} | Val R²: {r['val_r2']:.4f} | Test R²: {r['test_r2']:.4f}")


if __name__ == '__main__':
    main()

