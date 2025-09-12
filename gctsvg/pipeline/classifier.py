import torch

from torch_geometric.data import Data

def create_embedding_classifier_data(adata, graph_data, embedding_key = 'embedding', clustering_key = 'leiden'):
    # get embedding from adata_clusters
    embedding = adata.obsm[embedding_key]
    
    # get edge_index from graph_data
    edge_index = graph_data.edge_index

    # get edge_index from existing graph
    clustering = torchadata.obs[clustering_key].apply(int).to_numpy()
    clustering = torch.from_numpy(clustering)
    
    classifier_data = Data(x = feats, edge_index = edge_index, y = clustering)
    return classifier_data

class STClassification(object):
    def __init__(self, 
                 num_epochs,
                 project,
                 accelerator="auto",
                 devices=1,
                 strategy = "ddp",
                 num_nodes=1):
        self.num_epochs = num_epochs
        self.accelerator = accelerator
        self.devices = devices
        self.strategy = strategy
        self.num_nodes = num_nodes
        self.project = project
        self.output_dir = f"gctsvg_output/{project}"

        self.configure_callbacks()
        self.configure_trainer()

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            filename='{epoch}-{total loss:.4f}',
            monitor='total loss',
            save_top_k=10,
            mode='min',
            every_n_epochs=1,
            save_on_train_epoch_end=True,
            save_weights_only=True,
            auto_insert_metric_name=True)
        self.checkpoint_sync_callback = BestCheckpointSync()
    
    def configure_trainer(self):
        prediction_writer = EmbeddingWriter(output_dir=self.output_dir, write_interval="epoch")
        self.trainer = L.Trainer(accelerator=self.accelerator,
                 max_epochs = self.num_epochs, 
                 devices = self.devices,
                 strategy = self.strategy,
                 log_every_n_steps=1,
                 num_nodes=self.num_nodes,
                 default_root_dir=self.output_dir,
                 logger=loggers.CSVLogger(f'{self.output_dir}/logs', name='st_embedder'),
                 callbacks=[prediction_writer, PrintCallback(), self.checkpoint_callback, self.checkpoint_sync_callback])
    
    def train(self, ae_kwargs, datamodule):
        datamodule.setup("fit")

        ae = GCNDGIAE(**ae_kwargs)
        
        self.trainer.fit(model = ae, 
                     datamodule = datamodule)

        self.trainer.strategy.barrier("training end")        
        print(f"Training finished. Best model score: {self.checkpoint_callback.best_model_score}. Path: {self.checkpoint_callback.best_model_path}")
        # best_ae = GCNDGIAE.load_from_checkpoint(checkpoint_callback.best_model_path)
        # torch.save(best_ae.state_dict(), f"{self.output_dir}/gatae_model.ckpt")
        # trainer.save_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

    def compute_attributions(self, datamodule, embedder_ckpt_path, classifier_ckpt_path):
        datamodule.setup("predict")
        model = StackClassifier(embedder_ckpt_path, classifier_ckpt_path)

        # declare baseline
        # 
        
    
    def generate_embedding(self, datamodule):
        datamodule.setup("predict")

        # ae = GCNDGIAE.load_from_checkpoint(f"{self.output_dir}/gatae_model.ckpt")

        print("Predicting using best model...")
        best_ckpt_path = self.checkpoint_sync_callback.best_checkpoint_path
        self.trainer.predict(ckpt_path=best_ckpt_path, datamodule = datamodule, return_predictions=False)

        self.trainer.strategy.barrier("prediction end")