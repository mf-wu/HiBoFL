import numpy as np
import pandas as pd
from pymatgen.core import Structure
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital, IonProperty
from matminer.featurizers.structure import (
    SiteStatsFingerprint, StructuralHeterogeneity, ChemicalOrdering,
    StructureComposition, MaximumPackingEfficiency
)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


class SupervisedClassifierPipline:
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.df['structure'] = self.df['structure_json'].apply(lambda x: Structure.from_dict(x, fmt='json'))

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

    def prepare_classification_data(self):
        self.df_fea['k_class'] = (self.df_fea['k_ao'] <= 2).astype(int)
        drop_cols = [
            'nsites', 'nelements', 'formula_pretty', 'formula_anonymous', 'volume',
            'density', 'material_id', 'energy_per_atom', 'formation_energy_per_atom',
            'energy_above_hull', 'band_gap', 'is_metal', 'structure_json', 'label',
            'k_ao', 'k_log', 'structure', 'k_class'
        ]
        self.X = self.df_fea.drop(columns=drop_cols, errors='ignore')
        self.y = self.df_fea['k_class']
        return self.y.value_counts().to_dict()

    def compare_models(self): 
        base_models = [
            ("RandomForest", RandomForestClassifier),
            ("XGBoost", XGBClassifier),
            ("GradientBoosting", GradientBoostingClassifier),
            ("LightGBM", LGBMClassifier),
            ("DecisionTree", DecisionTreeClassifier),
            ("ExtraTrees", ExtraTreesClassifier),
            ("AdaBoost", AdaBoostClassifier),
            ("CatBoost", CatBoostClassifier)
        ]

        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=19981024)

        results = []
        for name, ModelClass in base_models:
            model = ModelClass(random_state=1998)
            scores = cross_val_score(model, self.X, self.y, cv=kfold, scoring='roc_auc')
            accu = cross_val_score(model, self.X, self.y, cv=kfold, scoring='accuracy')
            mean_roc = scores.mean()
            mean_acc = accu.mean()
            print(f"{name} - ROC AUC: {mean_roc}, Accuracy: {mean_acc}")
            results.append({
                'Model': name,
                'ROC AUC': mean_roc,
                'Accuracy': mean_acc
            })

        # SVC with scaling
        ss = StandardScaler()
        X_scaled = ss.fit_transform(self.X)
        clf = SVC(kernel='rbf', random_state=1998)
        scores = cross_val_score(clf, X_scaled, self.y, cv=kfold, scoring='roc_auc')
        accu = cross_val_score(clf, X_scaled, self.y, cv=kfold, scoring='accuracy')
        mean_roc = scores.mean()
        mean_acc = accu.mean()
        print(f"SVM - ROC AUC: {mean_roc}, Accuracy: {mean_acc}")
        results.append({
            'Model': 'SVM',
            'ROC AUC': mean_roc,
            'Accuracy': mean_acc
        })

        self.all_performance = pd.DataFrame(results)

    # Run several times to find the optimal feature numbers
    def feature_selection_wrapper(self, top_n_features, model=None): 
        if model is None:
            model = CatBoostClassifier(random_state=1998)

        model.fit(self.X, self.y)

        importance_set = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        self.selected_features = importance_set['feature'].head(top_n_features).tolist()
        self.X_selected = self.X[self.selected_features]

        # Refit with selected features
        scores = cross_val_score(
            CatBoostClassifier(random_state=1998, verbose=0),
            self.X_selected, self.y,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=19981024),
            scoring='roc_auc'
        )
        print(f"Ten-fold scores with top-{top_n_features} selected features via wrapper method: ROC_mean={np.mean(scores):.2f}")

    # We use 0.8 as the corr_threshold, top_n_features=11
    def feature_selection_pearson(self, top_n_features, corr_threshold, model=None):
        if model is None:
            model = CatBoostClassifier(random_state=1998)

        model.fit(self.X, self.y)
        importance_set = pd.DataFrame({
            'feature': self.X.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        selected_features = importance_set['feature'].head(top_n_features).tolist()
        X_s = self.X[selected_features].copy()

        model.fit(X_s, self.y)
        importance_set = pd.DataFrame({
            'feature': X_s.columns,
            'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=False).reset_index(drop=True)

        correlation_matrix = X_s.corr()
        highly_correlated_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > corr_threshold:
                    highly_correlated_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j]))

        for pair in highly_correlated_pairs:
            feature1, feature2 = pair
            if feature1 in X_s.columns and feature2 in X_s.columns:
                importance1 = importance_set.loc[importance_set['feature'] == feature1, 'importance'].values[0]
                importance2 = importance_set.loc[importance_set['feature'] == feature2, 'importance'].values[0]  
                if importance1 > importance2:
                    X_s = X_s.drop(columns=[feature2])
                else:
                    X_s = X_s.drop(columns=[feature1])

        self.final_selected_features = X_s.columns.tolist()
        self.X_final = X_s
        print(f"Selected {len(self.final_selected_features)} features after Pearson correlation filtering: {self.final_selected_features}")

    def export_final_dataset(self, output_path):
        df_final = pd.concat([self.df, self.X_final], axis=1)
        df_final['k_class'] = (df_final['k_ao'] <= 2).astype(int)
        df_final.drop(columns=['structure'], inplace=True, errors='ignore')
        df_final.to_json(output_path)
        self.df_final = df_final


if __name__ == "__main__":
    # json_path='./data/MP_dataset_C12_HTC.json'
    # output_path='./data/MP_C12_selected_features.json'
    pipeline = SupervisedClassifierPipline(json_path='./data/MP_dataset_C12_HTC.json')