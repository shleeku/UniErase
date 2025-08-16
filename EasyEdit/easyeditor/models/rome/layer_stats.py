import os
from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util.globals import *
from ...util.nethook import Trace, set_requires_grad
from ...util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="gpt2-xl", choices=["gpt2-xl", "EleutherAI/gpt-j-6B"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=100000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )


def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
    hparams=None
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        # Load_From_File
        # from datasets import Dataset
        # raw_ds = Dataset.from_file('XXX/XXX/wikipedia-train.arrow')
        # raw_ds = {'train': raw_ds}
        # raw_ds = load_dataset(
        #     ds_name,
        #     dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
        # )
        raw_ds = None
        if "wikipedia" in ds_name:
            # raw_ds = load_dataset(
            #     "wikipedia",
            #     # dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")["wikipedia"],
            #     dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")["wikipedia"],
            #     cache_dir="data/datasets/wikipedia"
            # ) # requires pip install datasets==2.13.1
            raw_ds = load_dataset(
                "Salesforce/wikitext",
                "wikitext-103-raw-v1",
                # split="train[:2%]",
                cache_dir="data/datasets/wikipedia"
            ) # datasets==4.0.0
# NO_PROXY="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" \
# no_proxy="localhost,127.0.0.1,.huggingface.co,.hf.co,cdn-lfs.hf.co" \
# HTTP_PROXY= \
# http_proxy= \
# HTTPS_PROXY= \
# https_proxy= \
# python run_edit.py
            
        if hasattr(model.config, 'n_positions'):
            maxlen = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            maxlen = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            maxlen = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            maxlen = model.config.seq_length
        else:
            raise NotImplementedError
                
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                maxlen = model.config.sliding_window or 4096
            else:
                maxlen = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            maxlen = 4096

        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens

        maxlen = 4096
        print(f"---{len(raw_ds['train'])}---")
        if "tofu-retain" in ds_name:
            print("Mixing...")
            if "tofu-retain90" in ds_name:
                temp = "retain90"
            if "tofu-retain95" in ds_name:
                temp = "retain95"
            if "tofu-retain99" in ds_name:
                temp = "retain99"
            retain_ds = load_dataset("locuslab/TOFU", temp,
                                     cache_dir="/data/datasets/tofu")

            def combine_qa(example):
                example["text"] = f"{example['question']} {example['answer']}"
                return example

            # 应用拼接函数到训练集
            retain_ds["train"] = retain_ds["train"].map(combine_qa)
            raw_ds["train"] = concatenate_datasets([retain_ds["train"], raw_ds["train"]])
        print(f"---{len(raw_ds['train'])}---")
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    # Continue with computation of statistics
    # batch_size = 100  # Examine this many dataset texts at once
    batch_size = 16
    if hasattr(model.config, 'n_positions'):
        npos = model.config.n_positions
    elif hasattr(model.config, 'max_sequence_length'):
        npos = model.config.max_sequence_length
    elif hasattr(model.config, 'max_position_embeddings'):
        npos = model.config.max_position_embeddings
    elif hasattr(model.config,'seq_length'):
        npos = model.config.seq_length
    else:
        raise NotImplementedError
        
    if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
        if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
            npos = model.config.sliding_window or 4096
        else:
            npos = 4096
    if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
            npos = 4096

    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        # model_name = model.config._name_or_path.replace("/", "_")
        model_name = model.config._name_or_path.rsplit("/")[-1]

    stats_dir = Path(stats_dir)
    file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    filename = stats_dir / file_extension

    print(f"Computing Cov locally....")

    ds = get_ds() if not filename.exists() else None

    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader, loader_type = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    if loader_type == "wrapped":
        with torch.no_grad():
            for batch_group in progress(loader, total=batch_count):
                for batch in batch_group:
                    batch = dict_to_(batch, f"cuda:{hparams.device}")
                    with Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                    ) as tr:
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
