import pickle
import threading

from pathlib import Path

class ProjectMetadata:
    _shared_state = {}
    _lock = threading.Lock()

    _SUBDIRS = {
        "clustering_dir": "clustering",
        "ct_data_dir": "ct_spatial_data",
        "plot_dir": "plots",
        "eval_dir": "evaluation",
        "domain_markers_dir": "domain_markers",
    }
    
    def __init__(self, project_folder=None):
        self.__dict__ = self._shared_state

        if "project_folder" not in self.__dict__:
            if project_folder is None:
                raise RuntimeError(
                    "Metadata not initialized. Use Metadata.create(project_folder) "
                    "or Metadata(project_folder)."                 
                )
            
            self.project_folder = Path(project_folder)
            self._load_if_needed()

    @classmethod
    def use(cls, project_folder):
        """Switch the active project.

        The metadata is Borg (process-wide `_shared_state`) and `__init__` only loads a
        project the first time — a later `ProjectMetadata(other)` is silently ignored, so
        the active project is sticky. Notebooks that analyze several projects in one kernel
        (e.g. tumor core vs interface) must call this to actually switch: it clears the
        shared state and reloads from disk. Returns the freshly bound instance.
        """
        with cls._lock:
            cls._shared_state.clear()
        return cls(project_folder)

    @property
    def dir_map(self):
        return self._SUBDIRS.copy()
    
    def _metadata_path(self):
        return self.project_folder / "project_metadata.pkl"

    def _load_if_needed(self):
        path = self._metadata_path()

        if path.exists():
            with open(path, "rb") as f:
                state = pickle.load(f)
                self.__dict__.update(state)
        else:
            raise RuntimeError(
                f"Metadata file {path} not found! Use Metadata.create(project_folder)."                
            )
            
    def __getattr__(self, name):
        if name in self._SUBDIRS:
            return self._ensure_dir(name)
        raise AttributeError(name)

    def _ensure_dir(self, name):
        folder = self._SUBDIRS[name]
        p = self.project_folder / folder
        p.mkdir(parents=True, exist_ok=True)

        setattr(self, name, p)
        return p
    
    @classmethod
    def create(cls,                
               project_folder: Path,
               adata_decomp,
               ct_key,
               overwrite = True):
        
        with cls._lock:
            project_folder.mkdir(parents=True, exist_ok=True)

            metadata_path = project_folder / "project_metadata.pkl"

            if metadata_path.exists() and not overwrite:
                raise RuntimeError("Metadata already exists!")

            # define metadata
            ct_list = sorted(list(adata_decomp.obs[ct_key].unique().astype(str)))    
            ct_mapping = {
                cell_type:i for i, cell_type in enumerate(ct_list) 
            }
            n_cell_types = len(ct_list)
    
            # create new instance
            obj = cls.__new__(cls)
            obj.__dict__ = cls._shared_state
            
            obj.project_folder = project_folder
            obj.ct_key = ct_key
            obj.ct_list = ct_list
            obj.ct_mapping = ct_mapping
            obj.n_cell_types = n_cell_types

            with open(metadata_path, "wb") as f:
                pickle.dump(obj.__dict__, f)

            return obj