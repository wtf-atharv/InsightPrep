import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from utils import preprocess_dataset,handle_imbalance
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA

# 🌐 Page Config
st.set_page_config(page_title="InsightPrep", layout="wide", page_icon="🎓")

st.title("📂 Advanced Dataset Profiler & Preprocessor")

# Maintain session state
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

# 📂 File Upload
uploaded_file = st.file_uploader("📤 Upload your CSV file", type=["csv"])
if uploaded_file:
    st.session_state.uploaded_df = pd.read_csv(uploaded_file)

# 🧠 Main Work Area
if st.session_state.uploaded_df is not None:
    df = st.session_state.uploaded_df
    df = df.apply(pd.to_numeric, errors='ignore')
    st.success("✅ File uploaded successfully!")
    st.dataframe(df.head())

    st.markdown("---")
    tabs = st.tabs(["📊 Profile Report", "⚙️ Preprocessing Options", "✅ Processed Output"])

    # --------------------- TAB 1: PROFILE REPORT ---------------------
    with tabs[0]:
        st.header("📋 Dataset Overview")

        # --- Basic Info ---
        st.subheader("📏 Basic Information")
        st.write(f"**Rows:** {df.shape[0]}  **Columns:** {df.shape[1]}")
        st.write(f"**Memory Usage:** {round(df.memory_usage(deep=True).sum() / 1024, 2)} KB")
        st.markdown("---")

        # --- Column Summary ---
        st.subheader("🔍 Column Details")
        info_df = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": df.notnull().sum().values,
            "Missing Count": df.isnull().sum().values,
            "Dtype": df.dtypes.values
        })
        st.dataframe(info_df, use_container_width=True)
        st.markdown("---")

        # --- Summary Statistics ---
        st.subheader("📊 Summary Statistics (Numeric Columns)")
        if len(df.select_dtypes(include=np.number).columns) > 0:
            st.dataframe(df.describe().T, use_container_width=True)
        else:
            st.info("No numeric columns available.")
        st.markdown("---")

        # --- Column-wise Visualization ---
        st.subheader("📈 Column Visualizations")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        st.caption("Automatically generated visualizations for each column in dataset.")
        cols = st.columns(2)
        for i, col in enumerate(df.columns):
            with cols[i % 2]:
                st.markdown(f"**🧩 {col}**")
                fig, ax = plt.subplots(figsize=(5, 3))
                if df[col].dtype in [np.int64, np.float64]:
                    sns.histplot(df[col], bins=25, kde=True, ax=ax)
                    ax.set_xlabel(col)
                else:
                    value_counts = df[col].value_counts().head(10)
                    sns.barplot(x=value_counts.values, y=value_counts.index, ax=ax)
                    ax.set_xlabel("Count")
                st.pyplot(fig)
        st.markdown("---")

        # --- Correlation Analysis ---
        st.subheader("🎯 Correlation Analysis")
        if len(numeric_cols) >= 2:
            df_encoded = df.copy()
            for col in df_encoded.columns:
                if df_encoded[col].dtype == 'object' or df_encoded[col].dtype.name == 'category':
                    try:
                        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
                    except Exception:
                        pass

            numeric_cols_encoded = df_encoded.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_encoded) >= 2:
                if st.checkbox("Show correlation heatmap (TEMP encoded)", value=False, key="heatmap_temp_encoded"):
                    corr_mat = df_encoded.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr_mat, cmap="coolwarm", annot=False, ax=ax)
                    st.pyplot(fig)

                target_col = st.selectbox("Select target column for correlation (TEMP encoding only)", options=list(df.columns))
                if target_col:
                    corr = df_encoded.corr(numeric_only=True)
                    if target_col not in corr.columns:
                        st.warning("⚠️ Selected target column not numeric or failed to encode.")
                    else:
                        target_corr = corr[target_col].drop(target_col, errors='ignore').abs().sort_values(ascending=False).head(10)
                        st.subheader(f"📊 Top 10 Correlated Features with `{target_col}`")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(x=target_corr.values, y=target_corr.index, ax=ax, palette="coolwarm")
                        ax.set_xlabel("Absolute Correlation")
                        ax.set_ylabel("Feature")
                        st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation analysis.")
        else:
            st.info("Not enough numeric columns for correlation analysis.")

    # --------------------- TAB 2: PREPROCESSING ---------------------
    with tabs[1]:
        st.header("⚙️ Data Preprocessing Settings")

        drop_cols = st.multiselect("Select columns to drop", options=df.columns)
        st.markdown("### 📉 Drop Low-Correlation Columns (Optional)")
        drop_low_corr = st.checkbox("Drop columns with low correlation relative to target column", key="drop_low_corr")
        corr_threshold = st.number_input("Set correlation threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

        target_col = st.selectbox("Select Target Column (to exclude from preprocessing)", options=["None"] + list(df.columns))
        if target_col == "None":
            target_col = None

        handle_missing = st.selectbox("Missing Value Handling", ["None", "Drop Rows", "Fill Values"])
        num_fill = None
        if handle_missing == "Fill Values":
            num_fill = st.selectbox("Numeric Fill Strategy", ["Mean", "Median"])
            st.caption("String columns automatically filled with mode.")

        remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True, key="remove_dupes")
        outlier_method = st.selectbox("Outlier Handling Method", ["None", "IQR Method", "Z-Score Method", "Isolation Forest"])
        normalization = st.selectbox("Normalization Technique", ["None", "Min-Max Scaler", "Standard Scaler", "Robust Scaler", "Z-Score (Custom)", "Log Transform"])
        encoding_type = st.selectbox("Encoding Technique", ["None", "Label Encoding", "One-Hot Encoding"])

        if st.button("🚀 Run All Preprocessing Steps", key="run_preprocess"):
            progress = st.progress(0)

            def update_progress(step, total):
                progress.progress(step / total)

            with st.spinner("Processing your dataset..."):
                df_processed = preprocess_dataset(
                    df=df,
                    drop_cols=drop_cols,
                    handle_missing=handle_missing,
                    num_fill=num_fill,
                    remove_duplicates=remove_duplicates,
                    outlier_method=outlier_method,
                    encoding_type=encoding_type,
                    normalization=normalization,
                    target_col=target_col,
                    progress_callback=update_progress
                )
                time.sleep(0.5)
                st.session_state.df_processed = df_processed
                st.session_state.target_col = target_col
                st.success("🎉 All preprocessing completed!")

    # --------------------- TAB 3: PROCESSED OUTPUT ---------------------
    with tabs[2]:
        if "df_processed" in st.session_state:
            df_processed = st.session_state.df_processed
            target_col = st.session_state.get("target_col", None)
            st.subheader("🧾 Processed Data Preview")
            st.dataframe(df_processed.head(), use_container_width=True)

            csv = df_processed.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download Processed Dataset", csv, "processed_dataset.csv", "text/csv")

            st.markdown("---")
            st.header("📈 Dataset Overview After Cleaning")

            st.subheader("📏 Basic Information")
            st.write(f"**Rows:** {df_processed.shape[0]}  **Columns:** {df_processed.shape[1]}")
            st.write(f"**Memory Usage:** {round(df_processed.memory_usage(deep=True).sum() / 1024, 2)} KB")
            st.markdown("---")

            # --- Column Summary ---
            st.subheader("🔍 Column Details")
            info_df = pd.DataFrame({
                "Column": df_processed.columns,
                "Non-Null Count": df_processed.notnull().sum().values,
                "Missing Count": df_processed.isnull().sum().values,
                "Dtype": df_processed.dtypes.values
            })
            st.dataframe(info_df, use_container_width=True)
            st.markdown("---")
            
            # --- Summary Statistics ---
            st.subheader("📊 Summary Statistics (Numeric Columns)")
            num_cols = df_processed.select_dtypes(include=np.number).columns
            if len(num_cols) > 0:
                st.dataframe(df_processed.describe().T, use_container_width=True)
            else:
                st.info("No numeric columns available.")

            st.markdown("---")

            # --- Correlation Analysis (After Cleaning) ---
            st.subheader("🎯 Correlation Analysis (After Cleaning)")
            df_encoded = df_processed.copy()


            numeric_cols_encoded = df_encoded.select_dtypes(include=np.number).columns.tolist()
            if len(numeric_cols_encoded) >= 2:
                if st.checkbox("Show correlation heatmap", value=False, key="heatmap_encoded"):
                    corr_mat = df_encoded.corr(numeric_only=True)
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.heatmap(corr_mat, cmap="coolwarm", annot=False, ax=ax)
                    st.pyplot(fig)

                if "target_col" in st.session_state and st.session_state.target_col:
                    target_col = st.session_state.target_col

                if target_col:
                    corr = df_encoded.corr(numeric_only=True)
                    if target_col not in corr.columns:
                        st.warning("⚠️ Selected target column not numeric or failed to encode.")
                    else:
                        target_corr = corr[target_col].drop(target_col, errors='ignore').abs().sort_values(ascending=False).head(10)
                        st.subheader(f"📊 Top 10 Correlated Features with `{target_col}`")
                        fig, ax = plt.subplots(figsize=(8, 5))
                        sns.barplot(x=target_corr.values, y=target_corr.index, ax=ax, palette="coolwarm")
                        ax.set_xlabel("Absolute Correlation")
                        ax.set_ylabel("Feature")
                        st.pyplot(fig)
            else:
                st.info("Not enough numeric columns for correlation analysis.")


            st.subheader("⚖️ Data Imbalance Handling")

            if "target_col" in st.session_state and st.session_state.target_col:
                target_col = st.session_state.target_col
            else:
                target_col = st.selectbox("Select Target Column", df_processed.columns)

            method = st.selectbox(
                "Choose imbalance handling method",
                ["None", "SMOTE Oversampling", "Random Undersampling"],
                key="imbalance_method"
            )

            if st.button("Apply Balancing"):
                df_balanced = handle_imbalance(df_processed, target_col, method)
                st.session_state["df_processed"] = df_balanced

            # --- PCA Integration ---
            st.subheader("🔍 PCA Dimensionality Reduction")

            if st.checkbox("Apply PCA", key="apply_pca"):
                if "target_col" in st.session_state and st.session_state.target_col:
                    target_col = st.session_state.target_col
                else:
                    target_col = None

                num_cols = df_processed.select_dtypes(include=np.number).columns.tolist()

                # ✅ Remove target column before PCA
                if target_col in num_cols:
                    num_cols.remove(target_col)

                if len(num_cols) < 2:
                    st.warning("⚠️ Not enough numeric features (excluding target) for PCA.")
                else:
                    n_components = st.slider(
                        "Select number of PCA components",
                        2,
                        min(len(num_cols), 10),
                        2,
                        key="pca_components"
                    )

                    try:
                        numeric_df = df_processed[num_cols].dropna()

                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=n_components)
                        pca_result = pca.fit_transform(numeric_df)

                        st.write("Explained Variance Ratio (%):", np.round(pca.explained_variance_ratio_ * 100, 2))

                        # --- 2D Scatter Plot of First 2 Components ---
                        if n_components >= 2:
                            fig, ax = plt.subplots()
                            if target_col and target_col in df_processed.columns:
                                sns.scatterplot(
                                    x=pca_result[:, 0],
                                    y=pca_result[:, 1],
                                    hue=df_processed[target_col],
                                    palette="viridis",
                                    s=60,
                                    edgecolor='k'
                                )
                                ax.legend(title=target_col, bbox_to_anchor=(1.05, 1), loc='upper left')
                            else:
                                sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], s=60, edgecolor='k')
                            ax.set_xlabel("Principal Component 1")
                            ax.set_ylabel("Principal Component 2")
                            st.pyplot(fig)

                        # --- PCA Dataframe ---
                        pca_df = pd.DataFrame(
                            pca_result,
                            columns=[f'PC{i+1}' for i in range(n_components)]
                        )
                        st.dataframe(pca_df)

                        # --- Cumulative Variance Plot ---
                        fig, ax = plt.subplots()
                        ax.plot(np.cumsum(pca.explained_variance_ratio_) * 100, marker='o')
                        ax.set_xlabel('Number of Components')
                        ax.set_ylabel('Cumulative Explained Variance (%)')
                        ax.set_title("PCA Cumulative Variance")
                        st.pyplot(fig)

                        # --- Correlation of PCA Components with Target ---
                        if target_col and target_col in df_processed.columns and df_processed[target_col].dtype in [np.int64, np.float64]:
                            combined_df = pd.concat([pca_df, df_processed[target_col].reset_index(drop=True)], axis=1)
                            corr_with_target = combined_df.corr(numeric_only=True)[target_col].drop(target_col, errors='ignore').sort_values(ascending=False)

                            st.subheader(f"📊 Correlation of PCA Components with `{target_col}`")
                            fig, ax = plt.subplots(figsize=(6, 4))
                            sns.barplot(x=corr_with_target.values, y=corr_with_target.index, palette="coolwarm", ax=ax)
                            ax.set_xlabel("Correlation with Target")
                            st.pyplot(fig)

                    except Exception as e:
                        st.error(f"⚠️ PCA failed: {e}")

                        # --- 📥 Download PCA Data with Target Column ---
                    if target_col and target_col in df_processed.columns:
                        pca_df[target_col] = df_processed[target_col].values

                    csv = pca_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download PCA Dataset (with target)",
                        data=csv,
                        file_name="pca_transformed.csv",
                        mime="text/csv",
                        key="download_pca"
                    )



else:
    st.info("👆 Upload a CSV file to begin profiling and preprocessing.")
