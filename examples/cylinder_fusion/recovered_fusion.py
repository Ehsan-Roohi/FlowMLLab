# -*- coding: utf-8 -*-
# 33fusion_fixed.py — Fusion-DeepONet (ensemble پنج‌مدلی)
# اصلاحات:
# 1) find_col(...) با آرگومان درست
# 2) dot_product: z[0] * z[1]
# 3) انتخاب خط سکون: absY درست + sort_values(['X','absY'])
# 4) axs[0].legend() بجای axs.legend()
# 5) فیلتر مقادیر خراب (|val|<1e10 و isfinite) برای جلوگیری از 1e30/1e31

from pathlib import Path
import argparse
from flowmllab.fusion_bundle import save_bundle, load_bundle, sha256
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dot, Add, Activation, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                              factor=0.2, # نرخ یادگیری را ۲ دهم برابر کن
                              patience=5,  # اگر ۵ اپاک بهتر نشد
                              min_lr=1e-6) # حداقل نرخ یادگیری
import matplotlib.pyplot as plt
import os
import re

# -----------------------------
# تنظیمات بصری برای خوانایی بهتر نمودارها
# -----------------------------
plt.rcParams.update({
    "figure.figsize": (12, 6),
    "figure.titlesize": 18,
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

# -----------------------------
# نام فایل‌ها
# -----------------------------
TRAIN_FILES = ['StructuredGridM=5.dat', 'StructuredGridM=7.dat', 'StructuredGridM=9.dat']
TEST_FILE   = 'StructuredGridM=10.dat'

# ستون‌های خروجی مورد انتظار
TARGET_COLS = ['Ma', 'TOV', 'P']

# -----------------------------
# Tecplot loader
# -----------------------------
def load_tecplot(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    variable_names, data_lines = [], []
    data_started, variables_started = False, False
    with open(filepath, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            u = s.upper()
            if u.startswith('VARIABLES'):
                variables_started = True
                if '=' in s:
                    rhs = s.split('=', 1)[1].strip()
                    vs = [v.strip().replace('"','').replace("'",'') for v in rhs.split(',') if v.strip()]
                    variable_names.extend(vs)
                    if len(vs) > 1:
                        variables_started = False
                continue
            if variables_started and not u.startswith('ZONE'):
                variable_names.append(s.replace('"','').replace("'",''))
                continue
            if u.startswith('ZONE'):
                variables_started, data_started = False, True
                continue
            if data_started:
                data_lines.append(s)
    if not variable_names:
        raise ValueError(f"No VARIABLES found in {filepath}")
    if not data_lines:
        raise ValueError(f"No data lines in {filepath}")
    all_vals = [float(v) for v in " ".join(data_lines).split()]
    ncol = len(variable_names)
    nrow = len(all_vals) // ncol
    arr = np.array(all_vals[:nrow*ncol]).reshape(nrow, ncol)
    return pd.DataFrame(arr, columns=variable_names)

def parse_mach_from_filename(filename: str) -> float:
    m = re.search(r'M=(\d+\.?\d*)', filename)
    if not m:
        raise ValueError(f"Cannot parse Mach from filename: {filename}")
    return float(m.group(1))

def find_col(df: pd.DataFrame, candidates) -> str:
    upper_map = {c.upper(): c for c in df.columns}
    for c in candidates:
        if c.upper() in upper_map:
            return upper_map[c.upper()]
    raise ValueError(f"None of {candidates} found. Columns: {list(df.columns)}")

# ==============================================================================
# DATA LOADING + FILTER (حذف 1e30/1e31 و مقدارهای خراب)
# ==============================================================================
def load_data_from_files(file_list, N_f=50000):
    frames = []
    for f in file_list:
        df = load_tecplot(f)
        df['Mach_input'] = parse_mach_from_filename(f)
        n_pick = min(N_f, len(df))
        df = df.sample(n=n_pick, random_state=42)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    # پیدا کردن نام صحیح ستون‌های x,y و خروجی‌ها
    x_col = find_col(data, ['x', 'X'])
    y_col = find_col(data, ['y', 'Y'])
    target_cols = [find_col(data, [c]) for c in TARGET_COLS]

    # فیلتر سفت‌وسخت: حذف اعداد خراب
    valid = np.ones(len(data), dtype=bool)
    for c in target_cols:
        valid &= np.isfinite(data[c]) & (np.abs(data[c]) < 1.0e10)
    valid &= np.isfinite(data[x_col]) & np.isfinite(data[y_col])
    data = data.loc[valid].copy()

    return data

# ==============================================================================
# MODEL ARCHITECTURE - FUSION-DEEPONET (با Activation بعد از Fusion)
# ==============================================================================
# ==============================================================================
# MODEL ARCHITECTURE - FUSION-DEEPONET (با Activation بعد از Fusion)
# ==============================================================================
def create_fusion_deeponet(branch_input_shape, trunk_input_shape, latent_dim=256, num_hidden_layers=4, dropout_rate=0.2):
    # Inputs
    branch_input = Input(shape=branch_input_shape, name='branch_input')
    trunk_input  = Input(shape=trunk_input_shape,  name='trunk_input')

    # Branch Network (BN -> Activation -> Dropout)
    branch_layers = []
    x = branch_input
    for _ in range(num_hidden_layers):
        x = Dense(latent_dim)(x)
        x = BatchNormalization()(x)         # <--- اضافه شد
        x = Activation('tanh')(x)
        x = Dropout(dropout_rate)(x)      # <--- اضافه شد
        branch_layers.append(x)
    branch_output = x

    # Trunk Network + Fusion Connections (BN -> Activation -> Dropout)
    y = trunk_input
    for i in range(num_hidden_layers):
        y_trunk_layer = Dense(latent_dim)(y)
        y_trunk_layer = BatchNormalization()(y_trunk_layer) # <--- اضافه شد
        y = Activation('tanh')(y_trunk_layer)           # لایه تنه
        
        y_fused = Add()([y, branch_layers[i]])            # مرحله همجوشی
        
        # پایدارسازی بعد از همجوشی
        y_stabilized = BatchNormalization()(y_fused)      # <--- اضافه شد
        y = Activation('tanh')(y_stabilized)
        y = Dropout(dropout_rate)(y)                    # <--- اضافه شد
        
    trunk_output = y

    # Combination (dot product) — درست
    dot_product = Dot(axes=1)([branch_output, trunk_output])

    # Map to 3 outputs
    final_output = Dense(3)(dot_product)

    model = Model(inputs=[branch_input, trunk_input], outputs=final_output)
    return model

# ==============================================================================
# TRAINING
# ==============================================================================
def train_ensemble(num_models=5, N_f=50000, epochs=500):
    MODEL_DIR.mkdir(parents=True, exist_ok=False)
    tf.keras.utils.set_random_seed(42)
    data = load_data_from_files(TRAIN_FILES, N_f=N_f)

    x_col = find_col(data, ['x', 'X'])
    y_col = find_col(data, ['y', 'Y'])
    target_cols = [find_col(data, [c]) for c in TARGET_COLS]

    X_branch = data[['Mach_input']].values
    X_trunk  = data[[x_col, y_col]].values
    Y        = data[target_cols].values

    scaler_branch = MinMaxScaler().fit(X_branch)
    scaler_trunk  = MinMaxScaler().fit(X_trunk)
    scaler_Y      = StandardScaler().fit(Y)

    X_branch_scaled = scaler_branch.transform(X_branch)
    X_trunk_scaled  = scaler_trunk.transform(X_trunk)
    Y_scaled        = scaler_Y.transform(Y)

    Xb_tr, Xb_va, Xt_tr, Xt_va, Y_tr, Y_va = train_test_split(
        X_branch_scaled, X_trunk_scaled, Y_scaled, test_size=0.2, random_state=42)

    early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
    reduce_lr      = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=40, min_lr=1e-6)

    def weighted_mse(y_true, y_pred):
        loss_ma  = tf.reduce_mean(tf.square(y_true[:, 0] - y_pred[:, 0]))
        loss_tov = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
        loss_p   = tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))
        return loss_ma + loss_tov + 5.0 * loss_p

    models = []
    for i in range(num_models):
        print(f"--- Training Fusion-DeepONet Model {i+1}/{num_models} ---")
        # افزایش ظرفیت: نورون‌های بیشتر و لایه‌های بیشتر
        model = create_fusion_deeponet(branch_input_shape=(1,), 
                                       trunk_input_shape=(2,),
                                       latent_dim=512,         # <-- افزایش از ۲۵۶ به ۵۱۲
                                       num_hidden_layers=6,    # <-- افزایش از ۴ به ۶
                                       dropout_rate=0.2)
        model.compile(optimizer=Adam(learning_rate=5e-4), loss=weighted_mse)

        model.fit([Xb_tr, Xt_tr], Y_tr,
                  validation_data=([Xb_va, Xt_va], Y_va),
                  epochs=epochs, batch_size=128,
                  callbacks=[early_stopping, reduce_lr],
                  verbose=1)

        model_path = str(MODEL_DIR / f'cylinder_fusion_model_{i}.keras')
        model.save(model_path)
        print(f"Model {i+1} saved to {model_path}")
        models.append(model)

    save_bundle(MODEL_DIR,
                [f'cylinder_fusion_model_{i}.keras' for i in range(num_models)],
                (scaler_branch, scaler_trunk, scaler_Y),
                {"source_sha256": sha256(__file__), "training_files":
                 {str(Path(f).resolve()): sha256(f) for f in TRAIN_FILES},
                 "sample_per_file": N_f, "sampling_seed": 42,
                 "sampling_before_filter": True, "epochs_limit": epochs,
                 "columns": [x_col, y_col, target_cols],
                 "tensorflow_version": tf.__version__,
                 "historical_paper_reproduction": False})
    return models, scaler_branch, scaler_trunk, scaler_Y, (x_col, y_col, target_cols)

# ==============================================================================
# UQ ALONG STAGNATION LINE
# ==============================================================================
def analyze_and_plot_stagnation_line_uq(num_models=5):
    manifest, (scaler_branch, scaler_trunk, scaler_Y) = load_bundle(MODEL_DIR)
    models = [tf.keras.models.load_model(MODEL_DIR / name, compile=False)
              for name in manifest['models']]
    test_df = load_tecplot(TEST_FILE)
    x_col_t = find_col(test_df, ['x', 'X'])
    y_col_t = find_col(test_df, ['y', 'Y'])
    target_cols_t = [find_col(test_df, [c]) for c in TARGET_COLS]

    # فیلتر روی تست
    valid = np.ones(len(test_df), dtype=bool)
    for c in target_cols_t:
        valid &= np.isfinite(test_df[c]) & (np.abs(test_df[c]) < 1.0e10)
    valid &= np.isfinite(test_df[x_col_t]) & np.isfinite(test_df[y_col_t])
    test_df = test_df.loc[valid].copy()

    # استخراج خط سکون: برای هر X نزدیک‌ترین نقطه به Y=0
    xy = test_df[[x_col_t, y_col_t]].values
    x_round = np.round(xy[:, 0], 12)
    df_center = pd.DataFrame({'X': x_round, 'Y': xy[:, 1], 'i': np.arange(len(xy))})
    idx_min = (df_center.assign(absY=lambda d: np.abs(d['Y']))
               .sort_values(['X','absY'])
               .groupby('X', as_index=False)
               .first()['i'].values)

    xy_stag = xy[idx_min]
    Y_true_stag = test_df.iloc[idx_min][target_cols_t].values

    order = np.argsort(xy_stag[:, 0])
    xy_stag = xy_stag[order]
    Y_true_stag = Y_true_stag[order]
    x_coords = xy_stag[:, 0]

    mach_to_predict = parse_mach_from_filename(TEST_FILE)
    mach_vec = np.full((len(x_coords), 1), mach_to_predict)
    mach_scaled  = scaler_branch.transform(mach_vec)
    trunk_scaled = scaler_trunk.transform(xy_stag)

    preds_scaled = []
    for m in models:
        p = m.predict([mach_scaled, trunk_scaled], verbose=0)
        preds_scaled.append(p)
    preds_scaled = np.stack(preds_scaled, axis=0)

    mean_scaled = preds_scaled.mean(axis=0)
    std_scaled  = preds_scaled.std(axis=0)

    mean_phys  = scaler_Y.inverse_transform(mean_scaled)
    lower_phys = scaler_Y.inverse_transform(mean_scaled - 2*std_scaled)
    upper_phys = scaler_Y.inverse_transform(mean_scaled + 2*std_scaled)

    labels = TARGET_COLS
    fig, axs = plt.subplots(3, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    for k, ax in enumerate(axs):
        ax.plot(x_coords, Y_true_stag[:, k], 'b.-', label='DSMC')
        ax.plot(x_coords, mean_phys[:, k], 'r.-', label='NN mean')
        ax.fill_between(x_coords, lower_phys[:, k], upper_phys[:, k], color='r', alpha=0.2, label='±2σ')
        ax.set_ylabel(labels[k])
        ax.grid(True, linestyle='--', linewidth=0.6)
    axs[0].legend()
    axs[-1].set_xlabel('x-coordinate (stagnation line)')
    plt.suptitle(f'Stagnation Line: {", ".join(TARGET_COLS)} with UQ (M={mach_to_predict})', y=0.97)
    os.makedirs('./figs', exist_ok=True)
    outpng = str(MODEL_DIR / 'stagnation.png')
    plt.savefig(outpng, dpi=300, bbox_inches='tight')
    print(f"Saved: {outpng}")
    np.savez_compressed(MODEL_DIR / 'predictions.npz', x=x_coords,
                        truth=Y_true_stag, mean=mean_phys,
                        lower=lower_phys, upper=upper_phys)
    plt.close(fig)

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recovered Fusion model; explicit training, frozen replay.")
    parser.add_argument('mode', choices=['train', 'replay'])
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('.'))
    parser.add_argument('--test-file', default=TEST_FILE)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--samples', type=int, default=50000)
    parser.add_argument('--members', type=int, default=5)
    args = parser.parse_args()
    MODEL_DIR = args.model_dir.resolve()
    TRAIN_FILES = [str(args.data_dir / f) for f in TRAIN_FILES]
    TEST_FILE = str(args.data_dir / args.test_file)
    if min(args.epochs, args.samples, args.members) < 1:
        parser.error('epochs, samples and members must be positive')
    if args.mode == 'train':
        train_ensemble(args.members, args.samples, args.epochs)
    else:
        analyze_and_plot_stagnation_line_uq()
