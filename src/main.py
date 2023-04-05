import torch
import numpy as np
import utility
import data
import model
import loss
from option import args
from trainer import Trainer
import os
from thop import profile
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']="1"
os.environ['CUDA_CACHE_PATH']='~/.cudacache'

# torch.manual_seed(args.seed)
torch.cuda.empty_cache()
checkpoint = utility.checkpoint(args)

def main():
    global model
    if args.data_test == ['video']:
        from videotester import VideoTester
        model = model.Model(args, checkpoint)
        t = VideoTester(args, model, checkpoint)
        t.test()
    else:
        if checkpoint.ok:
            loader = data.Data(args)
            _model = model.Model(args, checkpoint)
            # input = torch.randn(1, 3, 320, 180).cuda()
            # flops, params = profile(_model, inputs=(input, 0))
            # print("flops", str(flops / 1e9))
            # print("params", str(params / 1e6))
            _loss = loss.Loss(args, checkpoint) if not args.test_only else None
            t = Trainer(args, loader, _model, _loss, checkpoint)
            while not t.terminate():
                t.train()
                t.test()

            checkpoint.done()

if __name__ == '__main__':
    main()
