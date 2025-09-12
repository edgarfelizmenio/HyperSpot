import torch

from lightning.pytorch.callbacks import Callback

from ..model.ae import GCNDGIAE
from .embedding import STDataModule

class GCNDGIAEScorer(GCNDGIAE):
    def __init__(self,
                 ae,
                 Activation = "leakyrelu",
                 lr=1e-4):
        super(GCNDGIAEEmbedder, self).__init__()

    def forward(self, x, edge_index):
        pos_z = self.encoder(x, edge_index)
        pos_z, neg_z, summary = self.DGI(x, edge_index)
        dgi_loss = self.DGI.loss(pos_z, neg_z, summary)
        reconstruction = self.decoder(pos_z, edge_index)
        # return pos_z, reconstruction
        return pos_z, reconstruction, dgi_loss
        
    def test_step(self, batch, batch_idx):
        X = batch.x
        edge_index = batch.edge_index
        mask_type="node"
        
        # code, reconstruction, _ = self(X, edge_index)
        
        captum_model = to_captum_model(self)
        inputs, additional_forward_args = to_captum_input(X, edge_index, mask_type)
        print(X.shape, *X.shape)
        baselines = make_baseline_data(*X.shape)
        
        attrib_model = IntegratedGradients(captum_model)
        
        attributions, delta = attrib_model.attribute(inputs=inputs, baselines=baselines, additional_forward_args = additional_forward_args, return_convergence_delta=True)
        
        return attributions, delta
    
    def predict_step(self, batch, batch_idx):
        X =  batch.x
        edge_index = batch.edge_index
        code, reconstruction, _ = self(X, edge_index)
        # code, reconstruction = self(X, edge_index)
        node_ids = batch.n_id

        
        # only return predictions from seed nodes
        code = code[:batch.batch_size]
        reconstruction = reconstruction[:batch.batch_size]
        
        # also return node ids of seed for aggregation purposes
        node_ids = node_ids[:batch.batch_size]
        
        return node_ids, code, reconstruction
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-5)

class AttributionWriter(Callback):
    def __init__(self):
        super(AttributionWriter, self).__init__()

    def on_test_epoch_end(self, trainer, pl_module):
        if not self.interval.on_epoch:
            return
        epoch_batch_indices = trainer.test_loop.epoch_batch_indices
        

class STAttribution(object):
    def __init__(self,
                 project,
                 attribution="ig",
                 accelerator="auto",
                 devices=1,
                 strategy="ddp",
                 num_nodes=1):

        pass
    
    def compute_attributions(self, ckpt_path):
        datamodule.setup("test")
        self.trainer.test(ckpt_path=ckpt_path, datamodule = datamodule, return_predictions=False)
        self.trainer.strategy.barrier("attribution end")
        