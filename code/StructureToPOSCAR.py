# Export the structures in C1 and C2 for high-throughput calculations
# We employed the phonon-elasticity-thermal (PET) model proposed in our previous work within the HTC framework   
# https://pubs.acs.org/doi/10.1021/acs.jpca.2c06286

import pandas as pd
from pymatgen.core import Structure
import os


class StructureToPOSCAR:
    def __init__(self, json_path, structure_dir):
        self.json_path = json_path
        self.structure_dir = structure_dir
        self.df = pd.read_json(json_path)

    def export_poscars(self):
        cwd_home = os.getcwd()
        os.makedirs(self.structure_dir, exist_ok=True)

        for i in range(self.df.shape[0]):
            structure_id = self.df.loc[i, 'material_id']
            formula_id = self.df.loc[i, 'formula_pretty']
            poscar_name = f"POSCAR-conventional_standard-{formula_id}-{structure_id}.vasp"

            structure = Structure.from_dict(self.df.loc[i, 'structure_json'])

            os.chdir(self.structure_dir)
            structure.to(poscar_name, 'POSCAR')
            
            try:
                structure.to(poscar_name, fmt='poscar')
                print(f"Export: {poscar_name}")
            except Exception as e:
                print(f"Failed to export {poscar_name}: {e}")
            os.chdir(cwd_home)
            
                   
if __name__ == "__main__":
    strctopos = StructureToPOSCAR(json_path='./data/MP_dataset_cluster_0_1.json', structure_dir='./data/structures/')
    strctopos.export_poscars()