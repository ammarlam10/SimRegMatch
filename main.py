import torch
from itertools import product

from utils.args import SimRegMatch_parser
from tasks.SimRegMatchTrainer import SimRegMatchTrainer

def main():        
    parser = SimRegMatch_parser()
    args = parser.parse_args()
    # When using --gpus '"device=5"', GPU 5 appears as GPU 0 inside container
    # So we use GPU 0 if only one GPU is exposed, otherwise use args.gpu
    if torch.cuda.device_count() == 1:
        args.cuda = torch.device("cuda:0")
    else:
        args.cuda = torch.device(f"cuda:{args.gpu}")
    
    trainer = SimRegMatchTrainer(args)
    
    # Determine starting epoch (trainer may have updated it from checkpoint)
    start_epoch = args.start_epoch if args.resume else 1
    
    for epoch in range(start_epoch, args.epochs+1, 1):
        trainer.train(epoch)
        if epoch <= 5 or epoch % 10 == 0:
            trainer.validation(epoch)
    
    trainer.inference(epoch)


if __name__ == "__main__":
    main()
