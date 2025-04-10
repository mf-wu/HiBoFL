import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_parallel_coordinate, plot_slice, plot_optimization_history, \
    plot_contour, plot_param_importances
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve,
    precision_score, recall_score
)
from catboost import CatBoostClassifier
import os

class CatBoostOptimizer:
    def __init__(self, json_path):
        self.df = pd.read_json(json_path)
        self.X = self.df.drop([
            'nsites', 'nelements', 'formula_pretty', 'formula_anonymous', 'volume',
            'density', 'material_id', 'energy_per_atom', 'formation_energy_per_atom',
            'energy_above_hull', 'band_gap', 'is_metal', 'structure_json', 'label',
            'k_ao', 'k_log', 'k_class'
        ], axis=1)
        self.y = self.df['k_class'].values
        self.best_params = None
        self.model = None

    def objective(self, trial):
        param = {
            "random_state": 1998,
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "iterations": trial.suggest_int('iterations', 100, 1200),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-8, 10.0),
            "depth": trial.suggest_int("depth", 2, 10),
            "random_strength": trial.suggest_int("random_strength", 1, 10),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 10.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.05, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 50)
        }

        model = CatBoostClassifier(**param)
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=40, stratify=self.y)
        model.fit(X_train, Y_train)
        y_pred_test = model.predict_proba(X_test)[:, 1]
        return roc_auc_score(Y_test, y_pred_test)

    def optimize(self, n_trials=None):
        if n_trials is None:
            n_trials = 200
        study = optuna.create_study(sampler=TPESampler(), direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)
        
        print("Number of finished trials: {}".format(len(study.trials)))
        print("Best trial:")
        trial = study.best_trial
        print("   ROC AUC: {}".format(trial.value))
        print("   Params:")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value)) 

        self.best_params = {
            "random_state": 1998,
            **study.best_params
        }
        self.study = study 

    # contour_params=["depth", "learning_rate", "colsample_bylevel"]
    def plot_optimization_results(self, plot_types=None, contour_params=None):
        study = self.study
        if plot_types is None:
            plot_types = ["parallel", "importance", "slice", "contour", "history"]
        if "contour" in plot_types and not contour_params:
           raise ValueError("When using 'contour' in plot_types, you must specify 'contour_params'")

        plot_func_map = {
            "parallel": plot_parallel_coordinate,
            "importance": plot_param_importances,
            "slice": plot_slice,
            "contour": lambda study: plot_contour(study, params=contour_params),
            "history": plot_optimization_history
        }

        for ptype in plot_types:
            if ptype in plot_func_map:
                plot = plot_func_map[ptype](study)
                plot.update_layout(font_family="Arial")
                plot.show()
            else:
                print(f"Unknown plot type: {ptype}")

    # Our best optimized params used in the study:
    """
    params = {
    "random_state": 1998,
    "learning_rate": 0.04294897062505356,
    "iterations": 387,
    "l2_leaf_reg": 0.00015375354143706398,
    "depth": 6,
    "random_strength": 3,
    "bagging_temperature": 1.6076574305929292,
    "colsample_bylevel": 0.5636145604538867,
    "min_data_in_leaf": 37
    }
    """
    def train_best_model(self, params=None, save_model=None):
        if save_model is None:
            raise ValueError("You must specify whether to save the model by setting 'save_model=True' or 'False'.")
        params_to_use = params if params is not None else self.best_params
        assert params_to_use is not None, "No parameters provided. Run optimize() first."

        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X, self.y, test_size=0.1, random_state=2014, stratify=self.y)

        self.model = CatBoostClassifier(**params_to_use)
        self.model.fit(X_train, Y_train)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(Y_test, y_pred_proba)
        print(f"ROC AUC value (test): {auc:.2f}")

        self._plot_roc_curve(Y_test, y_pred_proba, auc)

        y_pred = self.model.predict(X_test)
        self._print_classification_metrics(Y_test, y_pred)

        if save_model:
            os.makedirs('./pretrained_model', exist_ok=True)
            with open('./pretrained_model/SupervisedPredictor.pkl', 'wb+') as f:
                pickle.dump(self.model, f)
            print("Model saved to ./pretrained_model/SupervisedPredictor.pkl")

    def _plot_roc_curve(self, y_true, y_score, auc_value):
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'AUC = {auc_value:.2f}', lw=1.8)
        plt.plot([0, 1], [0, 1], 'k--', c='darkgrey')
        plt.xlabel('False positive rate', fontsize=28, family='Arial')
        plt.ylabel('True positive rate', fontsize=28, family='Arial')
        plt.xlim(-0.03, 1)
        plt.ylim(0, 1.16)
        plt.xticks(fontproperties='Arial', size=22)
        plt.yticks(fontproperties='Arial', size=22)
        ax = plt.gca()
        bwith = 1.5
        for spine in ax.spines.values():
            spine.set_linewidth(bwith)
        plt.legend(prop={'family': 'Arial', 'size': 20})
        plt.tight_layout()
        plt.show()

    def _print_classification_metrics(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='weighted')
        rec = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")
        print(f"F Score: {f1:.4f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    # json_path='./data/MP_dataset_C12_HTC.json'
    catopt = CatBoostOptimizer(json_path='./data/MP_C12_selected_features.json')