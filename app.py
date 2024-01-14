from flask import Flask, render_template, request, send_file, redirect
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from sklearn.decomposition import PCA
import csv

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/clustering')
def clustering():
    return redirect('/index.html')

@app.route('/index.html', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Ambil file dataset yang diupload oleh pengguna
        file = request.files['dataset']
        # Baca dataset menggunakan Pandas
        df = pd.read_csv(file)

        # Ambil algoritma clustering yang dipilih oleh pengguna
        algorithm = request.form.get('algorithm')

        # Lakukan scaling pada fitur numerik
        scaler = StandardScaler()
        numeric_features = df.select_dtypes(include=['float64', 'int64'])
        df[numeric_features.columns] = scaler.fit_transform(numeric_features)

        # Inisialisasi variabel clusters
        clusters = None

        # Lakukan clustering berdasarkan algoritma yang dipilih
        if algorithm == 'kmeans':
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df)
            pca_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            print('Shape after PCA: ', pca_data.shape)
            print('Original shape: ', df.shape)
            print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca.explained_variance_ratio_)))

            model = joblib.load('model/kmeans_model.sav')  # Ganti angka cluster sesuai kebutuhan
            # Lakukan clustering pada dataset yang telah diubah skala
            clusters = model.fit_predict(df)

            plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=clusters, cmap='viridis')
            centroids_pca = pca.transform(model.cluster_centers_)
            plt.scatter(x=centroids_pca[:,0], y=centroids_pca[:,1], marker="o", s=500, linewidths=3, color="black")
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.title('Clustering Plot (PCA)')

            plt.savefig('static/clustering_plot.png')  # Simpan plot sebagai file gambar
            plt.close()
        elif algorithm == 'agglomerative':
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df)
            pca_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            print('Shape after PCA: ', pca_data.shape)
            print('Original shape: ', df.shape)
            print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca.explained_variance_ratio_)))

            model = joblib.load('model/agglomerative_wine__model_baru.sav')  # Ganti angka cluster sesuai kebutuhan
            # Lakukan clustering pada dataset yang telah diubah skala
            clusters = model.fit_predict(df)

            plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=clusters, cmap='viridis')

            centroids = []
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                cluster_points = pca_data[clusters == cluster]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)

            centroids = np.array(centroids)
            centroids_pca = centroids
                        
            plt.figure(figsize=(12, 10))
            plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=clusters, cmap="brg", s=40)
            plt.scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1], marker="o", s=500, linewidths=3, color="black")
            plt.title('Agglomerative Clustering of Data')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            plt.savefig('static/clustering_plot_agglo.png')  # Simpan plot sebagai file gambar
            plt.close()
        elif algorithm == 'gmm':
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(df)
            pca_data = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            print('Shape after PCA: ', pca_data.shape)
            print('Original shape: ', df.shape)
            print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca.explained_variance_ratio_)))

            # Load model GMM
            gmm_model = joblib.load('model/gmm_model.sav')

            # Lakukan clustering pada dataset yang telah diubah skala
            clusters = gmm_model.predict(df)

            # Visualisasi hasil clustering dengan GMM
            plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=clusters, cmap='viridis')

            centroids = []
            unique_clusters = np.unique(clusters)
            for cluster in unique_clusters:
                cluster_points = pca_data[clusters == cluster]
                centroid = cluster_points.mean(axis=0)
                centroids.append(centroid)

            centroids = np.array(centroids)
            centroids_pca = centroids
                        
            plt.figure(figsize=(12, 10))
            plt.scatter(pca_data.iloc[:, 0], pca_data.iloc[:, 1], c=clusters, cmap="brg", s=40)
            plt.scatter(x=centroids_pca[:, 0], y=centroids_pca[:, 1], marker="o", s=500, linewidths=3, color="black")
            plt.title('GMM')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')

            plt.savefig('static/gmm_clustering_plot.png')  # Simpan plot sebagai file gambar
            plt.close()

        return render_template('output.html', algorithm=algorithm, clusters=clusters)

    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    algorithm = request.form.get('algorithm')
    clusters = request.form.get('clusters')

    if clusters is not None:
        clusters = list(map(int, clusters.replace('\r\n', ' ').replace('[', '').replace(']', '').split()))

        # Buat dataframe dengan kolom 'Data Point' dan 'Cluster'
        result_df = pd.DataFrame({'Data Point': range(1, len(clusters) + 1), 'Cluster': clusters})

        # Simpan dataframe sebagai file CSV
        result_df.to_csv('clustering_results.csv', index=False, quoting=csv.QUOTE_NONNUMERIC)

        return send_file('clustering_results.csv', as_attachment=True)

    return 'No clustering results available for download.'

if __name__ == '__main__':
    app.run(debug=True)