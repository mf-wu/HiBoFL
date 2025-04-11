# HiBoFL
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
<div style="border:1px solid #ccc; border-radius:6px; padding:12px; background-color:#f9f9f9;">
<pre><code><span style="color:#a020f0;">@article</span>{<span style="color:#ba55d3;">wu2025hierarchy</span>,
  <span style="color:#003366;">title</span>={<span style="color:#003366;">Hierarchy-boosted funnel learning for identifying semiconductors with ultralow lattice thermal conductivity</span>},
  <span style="color:#003366;">author</span>={<span style="color:#003366;">Wu, Mengfan and Yan, Shenshen and Ren, Jie</span>},
  <span style="color:#003366;">journal</span>={<span style="color:#003366;">arXiv preprint arXiv:2501.06775</span>},
  <span style="color:#003366;">year</span>={<span style="color:#003366;">2025</span>}
}
</code></pre>
</div>
</details>
