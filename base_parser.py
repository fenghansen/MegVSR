import argparse

class BaseParser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument("--mode", default="train", choices=["train", "test"])
        self.parser.add_argument("--config", default="./config.yaml", help="path to config")
        self.parser.add_argument("--checkpoint", default='./saved_model/',help="path to checkpoint to restore")
        self.parser.add_argument("--model_name", "-n", default='VRRDB_5', type=str, help="model_name")
        self.parser.add_argument("--result_dir", default='./images/',help="path to result_dir to restore")
        self.parser.add_argument("--step_size", "-s", default=2, type=int, help="step_size")
        self.parser.add_argument("--batch_size", "-b", default=8, type=int, help="batch_size")
        self.parser.add_argument("--patch_size", "-p", default=32, type=int, help="patch_size")
        self.parser.add_argument("--crop_per_image", "-crop", default=4, type=int, help="crop_per_image")
        self.parser.add_argument("--learning_rate", "-lr", default=1e-4, type=float, help="learning_rate")
        self.parser.add_argument("--last_epoch", "-e", default=0, type=int, help="last_epoch")
        self.parser.add_argument("--stop_epoch", default=20, type=int, help="stop_epoch")
        self.parser.add_argument("--num_workers", "-w", default=4, type=int, help="num_of_workers")
        self.parser.add_argument("--nframes", "-f", default=5, type=int, help="nframes")
        return self.parser.parse_args()