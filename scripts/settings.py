DEBUG = False

def initialize():
    print("Loading settings")   
    import os
    
    #os.environ["TORCH_LOGS"] = "+dynamic"
    os.environ["DYNAMIC_SHAPE"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import torch
    import torch._dynamo as dt
    import random

    torch.manual_seed(23)
    random.seed(23)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32   = True

    torch._dynamo.config.cache_size_limit = 64

    torch._dynamo.config.dynamic_shapes = True
    torch._dynamo.config.automatic_dynamic_shapes = True
    
    #torch._dynamo.config.assume_static_by_default = True
    #torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True