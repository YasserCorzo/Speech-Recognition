import sys
import os
import configargparse
import random
import torch
import numpy as np

from trainer import Trainer


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def get_parser(parser=None, required=True):
    if parser is None:
        parser = configargparse.ArgumentParser(
            description="Train an automatic speech recognition (ASR) model",
            config_file_parser_class=configargparse.YAMLConfigFileParser,
            formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
        )

    parser.add(
        "--config",
        is_config_file=True,
        help="config file path",
        default="conf/base.yaml",
    )

    ## General utils
    parser.add_argument(
        "--tag", default="001", type=str, help="Experiment Tag for storing logs, models"
    )
    parser.add_argument("--seed", default=2023, type=int, help="Random seed")
    parser.add_argument(
        "--text_pad", default=-1, type=int, help="Padding Index for Text Labels"
    )
    parser.add_argument(
        "--audio_pad", default=0, type=int, help="Padding Index for Audio features"
    )

    ## I/O related
    parser.add_argument(
        "--train_json",
        type=str,
        default="data/train_sp/data_unigram300.json",
        help="Filename of train label data (json)",
    )
    parser.add_argument(
        "--valid_json",
        type=str,
        default="data/dev/data_unigram300.json",
        help="Filename of validation label data (json)",
    )
    parser.add_argument(
        "--dict",
        type=str,
        default="data/unigram300_units.txt",
        help="Filename of the dictionary/vocabulary file",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=".",
        help="Output Data Directory/Experiment Directory",
    )

    ## Encoder related
    parser.add_argument(
        "--idim", type=int, default=83, help="Input Feature Size (don't need to change)"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden size of Transformer encoder.",
    )
    parser.add_argument(
        "--attention_heads", type=int, default=4, help="Number of attention heads."
    )
    parser.add_argument(
        "--linear_units", type=int, default=1024, help="FFN intermediate size."
    )
    parser.add_argument(
        "--eblocks", type=int, default=12, help="Number of encoder layers."
    )
    parser.add_argument(
        "--edropout", type=float, default=0.1, help="Dropout rate for encoder."
    )

    ## Decoder related
    parser.add_argument(
        "--dblocks", type=int, default=6, help="Number of decoder layers."
    )
    parser.add_argument(
        "--ddropout", type=float, default=0.1, help="Dropout rate for decoder."
    )

    ## Batch related
    parser.add_argument(
        "--batch_bins",
        type=int,
        default=800000,
    )
    parser.add_argument(
        "--nworkers",
        dest="nworkers",
        type=int,
        default=1,
    )

    ## Optimization related
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--wdecay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--accum_grad", default=1, type=int, help="Number of gradient accumuration"
    )
    parser.add_argument(
        "--warmup_steps", default=25000, type=int, help="Number of lr warmup steps."
    )
    parser.add_argument(
        "--label_smoothing", default=0.1, type=float, help="Label smoothing rate."
    )

    ## Training config
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Path of model and optimizer to be loaded",
    )
    parser.add_argument(
        "--nepochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--ngpu",
        default=1,
        type=int,
        help="Number of GPUs. If not given, use all visible devices",
    )
    parser.add_argument(
        "--log_interval", default=200, type=int, help="Log interval in batches."
    )

    return parser


def main(cmd_args):
    ## Return the arguments from parser
    parser = get_parser()
    args, _ = parser.parse_known_args(cmd_args)

    ## Set Random Seed for Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    ## Set directories
    expdir = os.path.join(args.out_dir, "exp", "train_" + args.tag)
    model_dir = os.path.join(expdir, "ckpts")
    log_dir = os.path.join(expdir, "logs")
    tb_dir = os.path.join(expdir, "tensorboard")

    args.expdir = expdir
    args.model_dir = model_dir
    args.log_dir = log_dir
    args.tb_dir = tb_dir

    for x in [expdir, model_dir, log_dir, tb_dir]:
        os.makedirs(x, exist_ok=True)

    ## Load the token list
    ## We use the same token to represent <eos> and <sos>: last index
    char_list = ["<blank>"]  # blank for ctc is at 0 index
    with open(args.dict, "r", encoding="utf-8") as f:
        char_dict = [line.strip().split(" ")[0] for line in f.readlines()]
    char_list = char_list + char_dict + ["<eos>"]
    args.char_list = char_list
    args.odim = len(char_list)

    ## Start training
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])
