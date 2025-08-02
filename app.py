import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="🛍️ E-Commerce Customer Analysis", layout="wide")

st.title("🛍️ E-Commerce Customer Segmentation & Product Recommendation")

st.markdown("""
Upload a CSV file with the following columns:
- Recency (days since last purchase)
- Frequency (number of purchases)
- Monetary (total spend)
- Description (product description for recommendation)
""")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)

        required_cols = ['Recency', 'Frequency', 'Monetary', 'Description']
        if not all(col in df.columns for col in required_cols):
            st.error(f"CSV must contain columns: {required_cols}")
        else:
            # --- Customer Segmentation ---

            # Preview data
            st.subheader("Sample Data")
            st.dataframe(df.head())

            # Scale RFM
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(df[['Recency', 'Frequency', 'Monetary']])

            # PCA for 2D visualization
            pca = PCA(n_components=2, random_state=42)
            pca_components = pca.fit_transform(rfm_scaled)
            df['PCA1'] = pca_components[:, 0]
            df['PCA2'] = pca_components[:, 1]

            # KMeans Clustering
            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_labels = kmeans.fit_predict(df[['PCA1', 'PCA2']])

            cluster_name_map = {
                0: 'Loyal Customers',
                1: 'Potential Customers',
                2: 'Inactive Customers'
            }
            df['Cluster'] = [cluster_name_map[label] for label in cluster_labels]

            # Show scatter plot for segmentation
            st.subheader("📊 Customer Segments Visualization")
            fig, ax = plt.subplots(figsize=(8,6))
            sns.scatterplot(
                data=df,
                x='PCA1',
                y='PCA2',
                hue='Cluster',
                palette='Set2',
                s=100,
                ax=ax
            )
            ax.set_title("Customer Segmentation (PCA reduced)")
            st.pyplot(fig)

            # Show cluster counts
            st.subheader("Cluster Counts")
            st.write(df['Cluster'].value_counts().rename("Customer Count").reset_index().rename(columns={"index": "Segment"}))

            # Download segmented data
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Segmented Data CSV", data=csv_data, file_name="segmented_customers.csv", mime="text/csv")

            st.markdown("---")

            # --- Product Recommendation ---

            st.header("🔍 Product Recommendation")

            # Build TF-IDF matrix
            tfidf = TfidfVectorizer(stop_words='english')
            tfidf_matrix = tfidf.fit_transform(df['Description'].fillna(''))

            def recommend(product_name, top_n=5):
                product_name = product_name.lower()
                # Find matching indices (case-insensitive substring match)
                matching_idxs = df[df['Description'].str.lower().str.contains(product_name)].index

                if matching_idxs.empty:
                    return None
                idx = matching_idxs[0]
                cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                similar_indices = cosine_sim.argsort()[-(top_n+1):][::-1]
                similar_indices = [i for i in similar_indices if i != idx][:top_n]

                return df.loc[similar_indices, 'Description']

            # User input for product search
            product_input = st.text_input("Enter product name for recommendations")

            if product_input:
                recommendations = recommend(product_input)
                if recommendations is None or recommendations.empty:
                    st.warning("No similar products found.")
                else:
                    st.subheader(f"Recommendations for '{product_input}':")
                    for i, prod in enumerate(recommendations, 1):
                        st.write(f"{i}. {prod}")

    except Exception as e:
        st.error(f"Error processing file: {e}")

else:
    st.info("Please upload a CSV file to get started.")

