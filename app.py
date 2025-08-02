import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="📊 Customer Segmentation App", layout="centered")

st.title("📊 Customer Segmentation App")
st.markdown("Upload a CSV file with **Recency, Frequency, and Monetary** columns.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        rfm = pd.read_csv(uploaded_file)

        # Check for necessary columns
        required_cols = ['Recency', 'Frequency', 'Monetary']
        if not all(col in rfm.columns for col in required_cols):
            st.error("❌ The uploaded CSV must contain 'Recency', 'Frequency', and 'Monetary' columns.")
        else:
            # Display preview
            st.subheader("Preview of Uploaded Data")
            st.write(rfm.head())

            # Standard Scaling
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm[required_cols])

            # PCA for 2D reduction
            pca = PCA(n_components=2, random_state=42)
            rfm_pca = pca.fit_transform(rfm_scaled)

            # Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(rfm_pca)

            # Map cluster numbers to meaningful names
            cluster_name_map = {
                0: 'Loyal Customers',
                1: 'Potential Customers',
                2: 'Inactive Customers'
            }
            cluster_names = [cluster_name_map[label] for label in cluster_labels]

            # Combine everything into one DataFrame
            rfm['PCA1'] = rfm_pca[:, 0]
            rfm['PCA2'] = rfm_pca[:, 1]
            rfm['Cluster'] = cluster_names  # Use names instead of numbers

            # Show scatter plot
            st.subheader("📈 Customer Segments (Scatter Plot)")
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=rfm,
                x="PCA1", y="PCA2",
                hue="Cluster",
                palette="Set2",
                s=80,
                ax=ax
            )
            ax.set_title("Customer Segmentation using KMeans")
            st.pyplot(fig)

            # Show cluster counts
            st.subheader("📊 Cluster Counts")
            st.write(rfm['Cluster'].value_counts().rename("Customer Count").reset_index().rename(columns={"index": "Segment"}))

            # Download segmented data
            st.subheader("⬇️ Download Segmented Data")
            csv_download = rfm.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV with Segments",
                data=csv_download,
                file_name="segmented_customers.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")

