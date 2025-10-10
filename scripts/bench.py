from settings import initialize
initialize()

from src.mslm.benchmark.BLEU import main

def run(version: str, checkpoint: str, epoch: int):
    main(version, checkpoint, epoch)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a model using BLEU score.")
    parser.add_argument("--version", type=str, required=True, help="Model version to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint name to evaluate.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number of the checkpoint to evaluate.")

    args = parser.parse_args()

    run(args.version, args.checkpoint, args.epoch)
    

