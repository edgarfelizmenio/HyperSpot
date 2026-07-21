import torch

from pathlib import Path

from .metadata import ProjectMetadata

class STHypergraphData(object):
    def __init__(self,
                cell_feats,
                hyperedge_feats,
                cell_to_spot_id,
                cell_types,
                hyperedge_index,
                incidence_matrix,
                num_cell_types,
                num_spots):
        self._data = {
            "cell_feats": cell_feats,
            "hyperedge_feats": hyperedge_feats,
            "cell_to_spot_id": cell_to_spot_id,
            "cell_types": cell_types,
            "hyperedge_index": hyperedge_index,
            "incidence_matrix": incidence_matrix,
            "num_cell_types": num_cell_types,
            "num_spots": num_spots
        }

    def __getitem__(self, key):
        return self._data[key]
    
    def save(self):
        proj_metadata = ProjectMetadata()
        torch.save(self._data, proj_metadata.project_folder / "hypergraph.pth")
    
    @staticmethod
    def load():
        proj_metadata = ProjectMetadata()
        data = torch.load(proj_metadata.project_folder / "hypergraph.pth")
        return STHypergraphData(
            data["cell_feats"],
            data["hyperedge_feats"],
            data["cell_to_spot_id"],
            data["cell_types"],
            data["hyperedge_index"],
            data["incidence_matrix"],
            data["num_cell_types"],
            data["num_spots"]
        )


def get_ids_per_celltype(data: STHypergraphData):
    per_celltype_idx = [
        get_celltype_ids(data, ct) for ct in range(cell_types.max() + 1)
    ]
    
    return per_celltype_idx

def get_celltype_ids(data: STHypergraphData, ct: int):
    cell_types = data["cell_types"]
    return torch.where(cell_types == ct)[0]
