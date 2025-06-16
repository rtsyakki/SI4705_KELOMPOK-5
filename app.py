import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import io

# Page config
st.set_page_config(page_title="Reading Behavior Analysis Dashboard", layout="wide")

# Load data function
@st.cache_data
def load_data(filepath='reading_interest.csv'):
    df = pd.read_csv(filepath, sep=';')
    for col in ['Reading Frequency per week', 'Number of Readings per Quarter', 'Daily Reading Duration (in minutes)', 
                'Internet Access Frequency per Week', 'Daily Internet Duration (in minutes)', 
                'Tingkat Kegemaran Membaca (Reading Interest)']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

# Preprocess data function for model
@st.cache_data
def preprocess_data():
    df = load_data()

    # Fill missing values
    numerical_cols_with_nan = df.select_dtypes(include=np.number).columns[df.select_dtypes(include=np.number).isnull().any()].tolist()
    categorical_cols_with_nan = df.select_dtypes(include='object').columns[df.select_dtypes(include='object').isnull().any()].tolist()

    for col in numerical_cols_with_nan:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)

    for col in categorical_cols_with_nan:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)

    # Normalize numerical columns
    numerical_cols_to_scale = ['Reading Frequency per week', 'Number of Readings per Quarter',
                               'Daily Reading Duration (in minutes)', 'Internet Access Frequency per Week',
                               'Daily Internet Duration (in minutes)']

    scaler = MinMaxScaler()
    df[numerical_cols_to_scale] = scaler.fit_transform(df[numerical_cols_to_scale])

    # One-hot encoding
    categorical_cols = ['Provinsi', 'Year', 'Category']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Split features and target
    X = df_encoded.drop('Tingkat Kegemaran Membaca (Reading Interest)', axis=1)
    y = df_encoded['Tingkat Kegemaran Membaca (Reading Interest)']

    return df, df_encoded, X, y

# Load data
df = load_data()
df_processed, df_encoded, X, y = preprocess_data()

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Choose a section:",
    ["Data Overview", "Supervised Learning", "Unsupervised Learning"]
)

# Title
st.title("📊 Dashboard Analisis Perilaku Membaca")

if menu == "Data Overview":
    st.header("Data Overview")

    # EDA Section
    st.subheader("Exploratory Data Analysis")

    # Dataset Preview
    st.write("### Dataset Preview")
    st.write(df.head())

    # Dataset Info
    st.write("### Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    # Descriptive Statistics
    st.write("### Descriptive Statistics")
    st.write(df.describe())

    # Numerical Distributions
    st.write("### Numerical Columns Distribution")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    if 'Year' in numerical_cols:
        numerical_cols.remove('Year')

    num_cols = len(numerical_cols)
    if num_cols > 0:
        fig, axes = plt.subplots(num_cols, 1, figsize=(10, 5*num_cols))
        if num_cols == 1:
            axes = [axes]
        for i, col in enumerate(numerical_cols):
            sns.histplot(data=df, x=col, kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig)

    # Categorical Distributions
    st.write("### Categorical Columns Distribution")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        fig, ax = plt.subplots(len(categorical_cols), 1, figsize=(10, 5*len(categorical_cols)))
        if len(categorical_cols) == 1:
            ax = [ax]
        for i, col in enumerate(categorical_cols):
            sns.countplot(data=df, y=col, ax=ax[i])
            ax[i].set_title(f'Distribution of {col}')
        plt.tight_layout()
        st.pyplot(fig)

    # Correlation Matrix
    st.write("### Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot(fig)

elif menu == "Supervised Learning":
    st.header("Supervised Learning - Random Forest Prediction")

    try:
        with open('random_forest_model.pkl', 'rb') as file:
            model = pickle.load(file)
            st.success("✅ Model loaded successfully")

        # Input features
        st.subheader("Enter Features for Prediction")

        # Get pre-normalization statistics for proper scaling of inputs
        orig_df = load_data()

        # Numeric inputs with appropriate ranges based on original data
        numerical_input_cols = ['Reading Frequency per week', 'Number of Readings per Quarter',
                              'Daily Reading Duration (in minutes)', 'Internet Access Frequency per Week',
                              'Daily Internet Duration (in minutes)']

        input_data = {}

        # Create inputs with appropriate min/max values from original data
        for col in numerical_input_cols:
            min_val = float(orig_df[col].min())
            max_val = float(orig_df[col].max())
            default_val = float(orig_df[col].median())
            input_data[col] = st.slider(
                f"{col}:", 
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=0.1
            )

        # Categorical inputs
        categorical_input_cols = ['Provinsi', 'Year', 'Category']
        for col in categorical_input_cols:
            unique_values = sorted(orig_df[col].unique().tolist())
            input_data[col] = st.selectbox(f"{col}:", unique_values)

        if st.button("Predict"):
            # Create input DataFrame
            input_df = pd.DataFrame([input_data])

            # Apply the same preprocessing steps as in training
            # 1. Convert from comma to period decimal notation
            for col in numerical_input_cols:
                if isinstance(input_df[col].iloc[0], str):
                    input_df[col] = input_df[col].str.replace(',', '.').astype(float)

            # 2. Scale numerical features using the same scaler parameters
            numerical_scaler = MinMaxScaler()
            # First fit on original data to get the same scale parameters
            _ = numerical_scaler.fit(orig_df[numerical_input_cols])
            # Then transform our input data
            input_df[numerical_input_cols] = numerical_scaler.transform(input_df[numerical_input_cols])

            # 3. One-hot encode categorical features
            encoded_input = pd.get_dummies(input_df, columns=categorical_input_cols, drop_first=True)

            # 4. Align columns with training data
            missing_cols = set(X.columns) - set(encoded_input.columns)
            for col in missing_cols:
                encoded_input[col] = 0
            encoded_input = encoded_input[X.columns]

            # Make prediction
            prediction = model.predict(encoded_input)[0]

            # Display the prediction with appropriate context
            st.success(f"Predicted Reading Interest: {prediction:.2f}")

            # Provide interpretation
            if prediction > 7.5:
                st.info("📚 This indicates a high level of reading interest.")
            elif prediction > 5.0:
                st.info("📖 This indicates a moderate level of reading interest.")
            else:
                st.info("📝 This indicates a lower level of reading interest.")

            # Visualize feature importance for this prediction
            if st.checkbox("Show explanation of this prediction"):
                try:
                    import shap
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(encoded_input)

                    st.write("### Feature Contribution to Prediction")
                    # Convert to DataFrame for easier visualization
                    shap_df = pd.DataFrame({
                        'Feature': encoded_input.columns,
                        'Importance': np.abs(shap_values[0])
                    }).sort_values('Importance', ascending=False).head(10)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=shap_df)
                    plt.title('Top 10 Features Influencing This Prediction')
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not generate prediction explanation: {str(e)}")
                    st.write("To enable detailed prediction explanations, install SHAP: `pip install shap`")

    except FileNotFoundError:
        st.error("Model file not found. Please ensure random_forest_model.pkl exists in the current directory.")
    except Exception as e:
        st.error(f"Error in prediction: {str(e)}")
        st.write("Stack trace:", traceback.format_exc())

# In the else: block for Unsupervised Learning
else:  # Unsupervised Learning
    st.header("Unsupervised Learning - Clustering Analysis")

    # Clustering visualization
    try:
        # Use the already preprocessed data (no NaNs)
        scaler = StandardScaler()

        # First handle missing values explicitly
        X_cluster = df[['Reading Frequency per week', 'Number of Readings per Quarter',
                       'Daily Reading Duration (in minutes)', 'Internet Access Frequency per Week',
                       'Daily Internet Duration (in minutes)']].copy()

        # Impute missing values before standardization
        for col in X_cluster.columns:
            X_cluster[col] = X_cluster[col].fillna(X_cluster[col].median())

        # Now scale the data after missing values are handled
        X_scaled = scaler.fit_transform(X_cluster)

        # K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Plot clusters
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
        plt.scatter(pca.transform(kmeans.cluster_centers_)[:, 0],
                   pca.transform(kmeans.cluster_centers_)[:, 1],
                   marker='X', s=200, linewidths=3, color='red', label='Centroids')
        plt.title('Cluster Visualization')
        plt.legend()
        st.pyplot(fig)

        # Cluster analysis
        df_cluster = df.copy()
        df_cluster['Cluster'] = clusters + 1  # Adding 1 to match your cluster naming (1-based)

        cluster_labels = {
            1: "High Reading Enthusiasts",
            2: "Moderate Readers", 
            3: "Low Reading Activity"
        }

        # Map clusters to human-readable names
        df_cluster['Cluster Label'] = df_cluster['Cluster'].map(cluster_labels)

        # Display cluster distribution
        st.subheader("Cluster Distribution")
        cluster_counts = df_cluster['Cluster Label'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster Type', 'Count']
        st.write(cluster_counts)

        # Show cluster characteristics
        st.subheader("Cluster Characteristics")
        for cluster_num in range(1, 4):  # Clusters 1, 2, 3
            label = cluster_labels[cluster_num]
            st.write(f"\n**{label} (Cluster {cluster_num})**")
            cluster_data = df_cluster[df_cluster['Cluster'] == cluster_num]
            st.write(f"Number of readers: {len(cluster_data)} ({len(cluster_data)/len(df_cluster)*100:.1f}%)")

            # Get descriptive statistics for numerical columns in this cluster
            st.write("Statistical Summary:")
            st.write(cluster_data[X_cluster.columns].describe())

    except Exception as e:
        st.error(f"Error in clustering analysis: {str(e)}")

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Created with ❤️ using Streamlit")
