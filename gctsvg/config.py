from enum import Enum

# TODO:
# apply DDP to parallelize operations
# https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
# 
class AcceleratorOptions(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"