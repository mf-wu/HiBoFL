# HiBoFL
<a href="https://github.com/mf-wu/HiBoFL" target="_blank"><img src="https://img.shields.io/badge/Machine%20Learning-Lattice%20Thermal%20Conductivity-red.svg"></a>
![GitHub top language](https://img.shields.io/github/languages/top/mf-wu/HiBoFL)
![GitHub](https://img.shields.io/github/license/mf-wu/HiBoFL?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/mf-wu/HiBoFL?style=social&cacheSeconds=60) 

This repository is the implementation of the HiBoFL framework for the paper "*Hierarchy-boosted funnel learning for identifying semiconductors with ultralow lattice thermal conductivity*".  
- arXiv: [10.48550/arXiv.2501.06775](https://arxiv.org/abs/2501.06775)
- npj Computational Materials: [*npj Comput. Mater.* 11, 106 (2025)](https://www.nature.com/articles/s41524-025-01583-9#Sec10)
>Data-driven machine learning (ML) has demonstrated tremendous potential in material property predictions. However, the scarcity of materials data with costly property labels in the vast chemical space presents a significant challenge for ML in efficiently predicting properties and uncovering structure-property relationships. Here, we propose a novel hierarchy-boosted funnel learning (HiBoFL) framework, which is successfully applied to identify semiconductors with ultralow lattice thermal conductivity ($\kappa_\mathrm{L}$). By training on only a few hundred materials targeted by unsupervised learning from a pool of hundreds of thousands, we achieve efficient and interpretable supervised predictions of ultralow $\kappa_\mathrm{L}$, thereby circumventing large-scale brute-force ab initio calculations without clear objectives. As a result, we provide a list of candidates with ultralow $\kappa_\mathrm{L}$ for potential thermoelectric applications and discover a new factor that significantly influences structural anharmonicity. This HiBoFL framework offers a novel practical pathway for accelerating the discovery of functional materials.

### Schematic
![image](https://github.com/mf-wu/HiBoFL/blob/main/figure/Fig1.png)
*Hierarchy-boosted funnel learning (HiBoFL) framework for accelerating the discovery of functional materials.*

### Dependencies
```
env_name="HiBoFL"
conda create -n $env_name -y python=3.9.18
conda activate $env_name

# For crystal-related package
conda install -y -c conda-forge pymatgen
pip install matminer

# For machine learing package
pip install scikit-learn
pip install optuna
pip install xgboost
pip install lightgbm
pip install catboost

# For interpretable package
pip install shap
```

### How to cite
Welcome to cite our paper if you find our codes useful:
```bibtex
@article{wu2025hierarchy,
  title={Hierarchy-boosted funnel learning for identifying semiconductors with ultralow lattice thermal conductivity},
  author={Wu, Mengfan and Yan, Shenshen and Ren, Jie},
  journal={arXiv preprint arXiv:2501.06775},
  year={2025}
}
```
