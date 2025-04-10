import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import pickle
from sklearn.manifold import TSNE
from pymatgen.core import Structure
from pymatgen.analysis.local_env import VoronoiNN
from matminer.utils.caching import get_all_nearest_neighbors
from matminer.featurizers.utils.stats import PropertyStats
from shap.plots import _waterfall
plt.rcParams['font.family'] = 'Arial' 


class DataLoader:
    def __init__(self, json_path, model_path):
        self.df = pd.read_json(json_path)
        self.model = self.load_model(model_path)
        self.X, self.y = self.prepare_data()
        self.index = None

    def load_model(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def prepare_data(self):
        drop_cols = [
            'nsites', 'nelements', 'formula_pretty', 'formula_anonymous', 'volume',
            'density', 'material_id', 'energy_per_atom',
            'formation_energy_per_atom', 'energy_above_hull', 'band_gap',
            'is_metal', 'structure_json', 'label', 'k_ao', 'k_log', 'k_class'
        ]
        X = self.df.drop(columns=drop_cols, errors='ignore')
        y = self.df["k_class"].values
        return X, y
    
    # mp_id=["mp-1102254", "mp-1103379"]
    def query_by_mp_id(self, mp_id):
        if isinstance(mp_id, str):
            mp_id = [mp_id]

        results = []
        self.index = []  
        for mp in mp_id:
            idx = self.df.loc[self.df['material_id'] == mp].index
            if len(idx) == 0:
                print(f"mp_id {mp} not found.")
                continue
            index_val = idx[0]
            formula = self.df.loc[index_val, 'formula_pretty']
            label = self.y[index_val]
            results.append((mp, index_val, formula, label))
            self.index.append(index_val)

        if len(results) == 1:
            self.index = self.index[0]
            return results[0]
        return results


class SHAPAnalyzer:
    def __init__(self, model, X, y, df):
        self.df = df
        self.X = X
        self.y = y
        self.explainer = shap.Explainer(model)
        self.shap_values = self.explainer(X)
        self.shap_vals  = self.shap_values.values

    # cutoff = 1, cutoff=0.8
    def plot_clustering_bar(self, cutoff):
        clust = shap.utils.hclust(self.X, self.y, linkage="average")
        shap.plots.bar(self.shap_values, clustering=clust, clustering_cutoff=cutoff, show=False)
        fig = plt.gcf()
        fig.set_size_inches(5, 6)
        plt.show()

    def plot_summary_impact(self):
        shap.summary_plot(self.shap_values, self.X, show=False, cmap=plt.get_cmap("GnBu"))
        plt.gcf().set_size_inches(11.5, 6)
        plt.show()

    def plot_summary_bar(self):
        shap.summary_plot(self.shap_values, self.X, show=False, plot_type="bar")
        plt.gcf().set_size_inches(11.5, 6)
        plt.show()

    def plot_heatmap(self):
        shap.plots.heatmap(self.shap_values, show=False)
        plt.show()

    # highlight_indices=[499, 311]
    def plot_tsne(self, highlight_indices=None):
        shap_embedded = TSNE(n_components=2, perplexity=50, random_state=1998).fit_transform(self.shap_vals)
        shap_tsne = pd.DataFrame(shap_embedded)

        plt.figure(figsize=(6, 6))
        plt.scatter(
            shap_embedded[:, 0],
            shap_embedded[:, 1],
            c=self.shap_vals.sum(1).astype(np.float64),
            linewidth=0,
            alpha=1.0,
            cmap=plt.get_cmap("Blues"),
        )
        cb = plt.colorbar(aspect=30, orientation="horizontal")
        cb.set_alpha(1)
        cb.outline.set_linewidth(0)
        cb.ax.tick_params("x", length=0)
        cb.ax.xaxis.set_label_position("top")

        if highlight_indices is not None:
            if isinstance(highlight_indices, int):
                highlight_indices = [highlight_indices] 
            highlight_points = shap_tsne.loc[highlight_indices]
            plt.scatter(
                highlight_points.iloc[:, 0],
                highlight_points.iloc[:, 1],
                marker='o',
                color='b',
                s=150,
                edgecolors='blue',
                facecolors='none',
                linewidths=1.5
            )

        plt.axis("off")
        plt.show()

    def plot_individual_force(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        X_short = self.X.copy()
        original_columns = X_short.columns.tolist()
        new_columns = [f'f{i}' for i in range(len(X_short.columns))]
        X_short.columns = new_columns

        name_map = dict(zip(new_columns, original_columns[:len(new_columns)]))
        for short, full in name_map.items():
            print(f"{short:<3} → {full}")

        for i in idx:
            material_id = self.df.loc[i, 'material_id']
            formula = self.df.loc[i, 'formula_pretty']
            label = self.y[i]
            print(f"\n Index {i} | MP-ID: {material_id} | Formula: {formula} | Label: {label}")
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_vals[i],
                X_short.iloc[[i]],
                show=False,
                matplotlib=True,
                contribution_threshold=0.01,
                plot_cmap='GnPR'
            )
            plt.show()

    def plot_waterfall(self, idx):
        if isinstance(idx, int):
            idx = [idx]

        for i in idx:
            material_id = self.df.loc[i, 'material_id']
            formula = self.df.loc[i, 'formula_pretty']
            label = self.y[i]
            print(f"\n Index {i} | MP-ID: {material_id} | Formula: {formula} | Label: {label}")
            _waterfall.waterfall_legacy(
                self.explainer.expected_value,
                self.shap_vals[i]
            )
            plt.show()


class BondAnalyzer:
    def __init__(self, df):
        self.df = df
        self.df['structure'] = df['structure_json'].apply(lambda x: Structure.from_dict(x, fmt='json'))

    def cal_wa_bond_length(self, mp_id):
        strc = self.df.loc[self.df['material_id'] == mp_id, 'structure'].values[0]
        voro = VoronoiNN(extra_nn_info=True, weight="area")
        nns = get_all_nearest_neighbors(voro, strc)

        wa_bond_lengths = np.zeros((len(strc),))
        for i, nn in enumerate(nns):
            weights = [n["weight"] for n in nn]
            lengths = [n["poly_info"]["face_dist"] * 2 for n in nn]
            wa_bond_lengths[i] = PropertyStats.mean(lengths, weights)

        element_names = [site.species_string for site in strc]

        wabl = pd.DataFrame({
            "Element": element_names,
            "Weighted bond length": wa_bond_lengths
        })

        mean_val = wa_bond_lengths.mean()
        min_val = wa_bond_lengths.min()
        min_relative = min_val / mean_val
        
        summary = pd.DataFrame({
            "Element": ["mean", "min", "min relative"],
            "Weighted bond length": [mean_val, min_val, min_relative]
        })

        df_wa = pd.concat([wabl, summary], ignore_index=True)
        return df_wa
    

if __name__ == "__main__":
    # json_path='./data/MP_C12_selected_features.json'
    # model_path='./pretrained_model/SupervisedPredictor.pkl'
    data = DataLoader(json_path='./data/MP_C12_selected_features.json', model_path='./pretrained_model/SupervisedPredictor.pkl')