import torch
from pathlib import Path

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
    
    def save(self, project_results_dir: Path):
        torch.save(self._data, project_results_dir / "hypergraph.pth")
    
    @staticmethod
    def load(project_results_dir: Path):
        data = torch.load(project_results_dir / "hypergraph.pth")
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