import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (
    SiteStatsFingerprint, StructuralHeterogeneity, ChemicalOrdering,
    StructureComposition, MaximumPackingEfficiency
)


class StructureClusteringPipeline:
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.df['structure'] = self.df['structure_json'].apply(lambda x: Structure.from_dict(x, fmt='json'))
        self.X_pre = None
        self.X_pca = None
        self.predictions = None
        self.df_fea = None

    def featurize(self):
        featurizer = MultipleFeaturizer([
            SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"),
            StructuralHeterogeneity(),
            ChemicalOrdering(),
            MaximumPackingEfficiency(),
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
            StructureComposition(Stoichiometry()),
            StructureComposition(ElementProperty.from_preset("magpie")),
            StructureComposition(ValenceOrbital(props=['frac'])),
            StructureComposition(IonProperty(fast=True))
        ])
        self.df_fea = featurizer.featurize_dataframe(self.df, col_id="structure")

    def preprocess(self, save_model=None):
        if save_model is None:
            raise ValueError("You must specify whether to save the model by setting 'save_model=True' or 'False'.")
        
        drop_cols = ['nsites', 'nelements', 'formula_pretty', 'formula_anonymous', 'volume',
                     'density', 'material_id', 'energy_per_atom', 'formation_energy_per_atom',
                     'energy_above_hull', 'band_gap', 'is_metal', 'bulk_modulus', 'shear_modulus',
                     'structure_json', 'structure']
        X = self.df_fea.drop(columns=drop_cols)
        ss = StandardScaler()
        qt = QuantileTransformer(output_distribution='normal', random_state=1998)
        X_ss = ss.fit_transform(X)
        X_qt = qt.fit_transform(X_ss)
        self.X_pre = X_qt
        
        if save_model:
            os.makedirs('./pretrained_model', exist_ok=True)
            with open('./pretrained_model/PreSS.pkl', 'wb+') as f:
                pickle.dump(ss, f)
            print("StandardScaler model saved to ./pretrained_model/PreSS.pkl")
            with open('./pretrained_model/PreQT.pkl', 'wb+') as f:
                pickle.dump(qt, f)
            print("QuantileTransformer model saved to ./pretrained_model/PreQT.pkl")

    # X = X_pre
    def determine_pca_components(self, X):
        pca = PCA(random_state=19981024, n_components=X.shape[1])
        pca.fit(X)
        evr = np.cumsum(pca.explained_variance_ratio_)
        for i, a in enumerate(evr):
            if a <= 1:
                print('{:<20d}{:<80f}'.format(i+1, a))
        self._plot_pca_variance(evr)
        return evr

    def _plot_pca_variance(self, evr):
        plt.rc('font', family='Arial')
        fig, ax1 = plt.subplots(figsize=(7, 5))
        ax1.set_xlabel('Number of principal components', fontsize=24)
        ax1.set_ylabel('Total explained variance', fontsize=24)
        ax1.xaxis.set_major_locator(MultipleLocator(30))
        ax1.tick_params(axis='both', labelsize=18, direction='out', width=1)
        plt.xlim(-15, 280)
        plt.ylim(0.1, 1.1)
        plt.vlines(83, ymin=0, ymax=0.99, linewidth=1.2, colors='b', linestyles="--")
        plt.hlines(0.99, xmin=-15, xmax=83, linewidth=1.2, colors='b', linestyles="--")
        ax1.plot(evr, linestyle='-', linewidth=1.8, color='steelblue')
        ax1.plot(83, evr[83], marker='.', markersize=18, markeredgecolor='white', color='b')
        for spine in ax1.spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

    # X = X_pre, n_components=83
    def reduce_dimension(self, X, n_components, save_model=None):
        if save_model is None:
            raise ValueError("You must specify whether to save the model by setting 'save_model=True' or 'False'.")
        pca = PCA(random_state=19981024, n_components=n_components)
        self.X_pca = pca.fit_transform(X)
        if save_model:
            os.makedirs('./pretrained_model', exist_ok=True)
            with open('./pretrained_model/PCA.pkl', 'wb+') as f:
                pickle.dump(pca, f)
            print("PCA model saved to ./pretrained_model/PCA.pkl")

    # X = X_pca
    def sum_squared_errors(self, X):
        SSE = []
        for k in range(4, 16):
            estimator = KMeans(random_state=1998, n_init="auto", n_clusters=k)
            estimator.fit(X)
            SSE.append(estimator.inertia_)
        plt.figure(figsize=(7, 5))
        plt.plot(range(4, 16), SSE, marker='.', markersize=18, linewidth=1.5)
        plt.plot(7, SSE[3], marker='.', markersize=18, color='b')
        plt.xlabel('Number of clusters', fontsize=24)
        plt.ylabel('Sum of squared errors (10$^6$)', fontsize=24)
        plt.xticks(size=18)
        plt.yticks(size=18)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

    # X = X_pca
    def silhouette_analysis(self, X):
        scores = []
        for k in range(4, 16):
            estimator = KMeans(random_state=1998, n_init="auto", n_clusters=k)
            estimator.fit(X)
            scores.append(silhouette_score(X, estimator.labels_))
        plt.figure(figsize=(8, 5))
        plt.bar(range(4, 16), scores, color='#97C8AF', alpha=0.6, width=0.7)
        plt.xlabel('Number of clusters', fontsize=32)
        plt.ylabel('Silhouette coefficient', fontsize=32)
        plt.xticks(size=28)
        plt.yticks(size=28)
        plt.ylim(0.09, 0.19)
        for spine in plt.gca().spines.values():
            spine.set_linewidth(1.5)
        plt.tight_layout()
        plt.show()

    # X = X_pca
    def cluster_and_visualize(self, X, n_clusters, save_model=None):
        if save_model is None:
            raise ValueError("You must specify whether to save the model by setting 'save_model=True' or 'False'.") 
        model = KMeans(random_state=1998, n_init="auto", n_clusters=n_clusters)
        self.predictions = model.fit_predict(X)
        self.df_fea['label'] = self.predictions

        if save_model:
            os.makedirs('./pretrained_model', exist_ok=True)
            with open('./pretrained_model/UnsupervisedPredictor.pkl', 'wb+') as f:
                pickle.dump(model, f)
            print("K-means model saved to ./pretrained_model/UnsupervisedPredictor.pkl")

        tsne = TSNE(n_components=2, random_state=1998)
        X_tsne = tsne.fit_transform(X)
        X_tsne = pd.DataFrame(X_tsne, columns=['TSNE-1', 'TSNE-2'])
        X_tsne['label'] = self.predictions
        X_tsne['material_id'] = self.df['material_id']
        X_tsne['formula_pretty'] = self.df['formula_pretty']

        plt.figure(figsize=(8, 7))
        palette = ['#C59D94', "#FAC795", "#FF9896", "#9EC4BE", "#619DB8", "#A4C8D9", "#48C0AA"]
        for i in range(n_clusters):
            Xi = X_tsne[X_tsne['label'] == i]
            plt.scatter(Xi['TSNE-1'], Xi['TSNE-2'], color=palette[i], s=8, label=f"C$_{{{i+1}}}$")
        plt.axis('off')
        plt.legend(bbox_to_anchor=(-0.2, 0.65), loc=3, frameon=False, fontsize=16)
        plt.tight_layout()
        plt.show()

    def merge_clusters(self, labels, output_path, drop_columns=['structure']):
        filtered_dfs = []
        for label in labels:
            if label in self.df_fea['label'].values:
                df_label = self.df_fea[self.df_fea['label'] == label].reset_index(drop=True)
                print(f"Label {label} number: {df_label.shape}")
                filtered_dfs.append(df_label)
            else:
                print(f"Warning: Label {label} not found in the data.")

        if filtered_dfs:
            df_combined = pd.concat(filtered_dfs, axis=0).reset_index(drop=True)
            df_final = df_combined.drop(columns=drop_columns)
            # save as JSON file
            df_final.to_json(output_path)
            self.df_final = df_final 
            print(f"Data saved in {output_path}")
            return df_final
        else:
            print("No data to merge.")
            return None


if __name__ == "__main__":
    # json_path='./data/MP_dataset_screened.json'
    # output_path='./data/MP_dataset_cluster_0_1.json'
    pipeline = StructureClusteringPipeline(json_path='./data/MP_dataset_screened.json')