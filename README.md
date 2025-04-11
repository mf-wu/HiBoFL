# HiBoFL
<a href="https://github.com/mf-wu/HiBoFL" target="_blank"><img src="https://img.shields.io/badge/Github-document_classfication-red.svg"></a>

<a href="https://github.com/mf-wu/HiBoFL">
  <img alt="HiBoFL" src="https://img.shields.io/badge/Machine%20Learning-Lattice Thermal Conductivity-blue.svg">
</a> 

![image](https://img.shields.io/badge/Release-Ver1.0.0-blue.svg)
![GitHub](https://img.shields.io/github/license/mf-wu/HiBoFL?style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/mf-wu/HiBoFL)
![GitHub issues](https://img.shields.io/github/issues-raw/mf-wu/HiBoFL?style=flat-square)
![GitHub closed issues](https://img.shields.io/github/issues-closed/mf-wu/HiBoFL?style=flat-square)
![GitHub Repo stars](https://img.shields.io/github/stars/mf-wu/HiBoFL?style=social) 

### Description
This repository is the implementation of the HiBoFL framework for the paper "*Hierarchy-boosted funnel learning for identifying semiconductors with ultralow lattice thermal conductivity*" published in *npj Computational Materials*.  
arXiv: [10.48550/arXiv.2501.06775](https://arxiv.org/abs/2501.06775)

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
```
@article{wu2025hierarchy,
  title={Hierarchy-boosted funnel learning for identifying semiconductors with ultralow lattice thermal conductivity},
  author={Wu, Mengfan and Yan, Shenshen and Ren, Jie},
  journal={arXiv preprint arXiv:2501.06775},
  year={2025}
}
```
