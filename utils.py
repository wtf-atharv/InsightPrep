import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


def handle_missing_values(df, strategy, num_fill=None):
    df = df.copy()
    if strategy == "Drop Rows":
        df.dropna(inplace=True)
    elif strategy == "Fill Values":
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                if num_fill == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
    return df


def handle_outliers(df, method):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    if method == "IQR Method":
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[numeric_cols] < (Q1 - 1.5 * IQR)) |
                  (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]
    elif method == "Z-Score Method":
        z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
        df = df[(z < 3).all(axis=1)]
    elif method == "Isolation Forest":
        iso = IsolationForest(contamination=0.05, random_state=42)
        preds = iso.fit_predict(df[numeric_cols])
        df = df[preds == 1]
    return df


def encode_data(df, encoding_type):
    df = df.copy()
    if encoding_type == "Label Encoding":
        label_enc = LabelEncoder()
        for col in df.select_dtypes(include="object").columns:
            df[col] = label_enc.fit_transform(df[col].astype(str))
    elif encoding_type == "One-Hot Encoding":
        df = pd.get_dummies(df, drop_first=True)
    return df


def normalize_data(df, normalization):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns

    if normalization == "Min-Max Scaler":
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif normalization == "Standard Scaler":
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif normalization == "Robust Scaler":
        scaler = RobustScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    elif normalization == "Z-Score (Custom)":
        df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()
    elif normalization == "Log Transform":
        df[numeric_cols] = np.log1p(df[numeric_cols] - df[numeric_cols].min() + 1)

    return df


def preprocess_dataset(df, drop_cols, handle_missing, num_fill,
                       remove_duplicates, outlier_method,
                       encoding_type, normalization,
                       target_col=None, progress_callback=None):

    df_processed = df.copy()
    total_steps = 7
    step = 0

    st.info("🚀 Preprocessing started...")

    # --- Drop Columns ---
    if drop_cols:
        existing_cols = [col for col in drop_cols if col in df_processed.columns]
        if existing_cols:
            df_processed.drop(columns=existing_cols, inplace=True)
            st.warning(f"🗑️ Dropped {len(existing_cols)} column(s): {', '.join(existing_cols)}")
    else:
        st.info("✅ No columns dropped.")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    # --- Handle Missing Values ---
    before_na = df_processed.isna().sum().sum()
    df_processed = handle_missing_values(df_processed, handle_missing, num_fill)
    after_na = df_processed.isna().sum().sum()
    filled = before_na - after_na
    st.success(f"🔧 Missing values handled: {filled} filled.")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    # --- Remove Duplicates ---
    if remove_duplicates:
        before_rows = len(df_processed)
        df_processed.drop_duplicates(inplace=True)
        removed = before_rows - len(df_processed)
        st.warning(f"♻️ {removed} duplicate rows removed.")
    else:
        st.info("✅ No duplicate removal applied.")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    # --- Handle Outliers ---
    before_rows = len(df_processed)
    df_processed = handle_outliers(df_processed, outlier_method)
    after_rows = len(df_processed)
    removed_outliers = before_rows - after_rows
    st.warning(f"🚫 {removed_outliers} outlier rows removed using {outlier_method or 'none'}.")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    # --- Normalization (Skip Target Column) ---
    if normalization:
        cols_to_normalize = [c for c in df_processed.select_dtypes(include=['int64', 'float64']).columns 
                             if c != target_col]
        if cols_to_normalize:
            df_processed[cols_to_normalize] = normalize_data(df_processed[cols_to_normalize], normalization)
            st.success(f"📊 Normalized {len(cols_to_normalize)} numeric column(s) using {normalization}.")
        else:
            st.info("ℹ️ No numeric columns found for normalization.")
    else:
        st.info("✅ Normalization skipped.")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    # --- Encoding ---
    before_cols = df_processed.shape[1]
    df_processed = encode_data(df_processed, encoding_type)
    after_cols = df_processed.shape[1]
    added_cols = after_cols - before_cols
    st.success(f"🔡 Encoding applied using {encoding_type} ({added_cols} new column(s) added).")
    step += 1
    if progress_callback: progress_callback(step, total_steps)

    st.success("🎯 Preprocessing complete!")
    if progress_callback: progress_callback(total_steps, total_steps)

    return df_processed


def handle_imbalance(df, target_col, method="None"):
    st.subheader("⚖️ Data Imbalance Handling")

    if not target_col or target_col not in df.columns:
        st.warning("⚠️ Please select a valid target column before applying SMOTE or undersampling.")
        return df

    if df[target_col].nunique() <= 1:
        st.warning("⚠️ Target column must have at least two classes for sampling.")
        return df

    X = df.drop(columns=[target_col])
    y = df[target_col]

    st.write("📊 Class Distribution Before:")
    st.bar_chart(y.value_counts())

    if method == "SMOTE Oversampling":
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        st.success("✅ Applied SMOTE Oversampling (balanced minority class).")

    elif method == "Random Undersampling":
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        st.success("✅ Applied Random Undersampling (reduced majority class).")

    else:
        st.info("ℹ️ No sampling method applied.")
        return df

    df_resampled = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=target_col)], axis=1)

    st.write("📊 Class Distribution After:")
    st.bar_chart(df_resampled[target_col].value_counts())

    # --- Download Button ---
    csv = df_resampled.to_csv(index=False).encode('utf-8')
    st.download_button(
        label=f"📥 Download Resampled Dataset ({method})",
        data=csv,
        file_name=f"resampled_{method.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key="download_resampled"
    )

    return df_resampled
