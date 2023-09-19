# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import argparse, os
import numpy as np

import torch.nn.utils.prune as prune

from llama import Llama
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)

def EnsureDirExists(dir):
    if not os.path.exists(dir):
        print("Creating %s" % dir)
        os.makedirs(dir)

def main(
    tensor_dir: str,
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    model = generator.model

    for n, m in model.named_modules():
        # export linear
        if isinstance(m, ColumnParallelLinear) or isinstance(m, RowParallelLinear):
            #print(n, m, m.weight.shape)
            #prune.l1_unstructured(m, name="weight", amount=0.0)
            weight = m.weight.detach().cpu().numpy()
            np.save(f"{tensor_dir}/{n}.npy", weight)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='llama2 export')

    parser.add_argument('--ckpt_dir', default='llama-2-7b', help='The directory containing checkpoint files for the pretrained model.')
    parser.add_argument('--tokenizer_path', default='tokenizer.model', help='The path to the tokenizer model used for text encoding/decoding.')
    parser.add_argument('--max_seq_len', default='128', help='The maximum sequence length for input prompts. Defaults to 128.')
    parser.add_argument('--max_batch_size', default='4', help='The maximum batch size for generating sequences. Defaults to 4.')

    args = parser.parse_args()

    tensor_dir = f"/scratch/yifany/sconv/inputs/{args.ckpt_dir}"

    main(tensor_dir, args.ckpt_dir, args.tokenizer_path, args.max_seq_len, args.max_batch_size)
