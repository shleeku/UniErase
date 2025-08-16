from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .AlphaEdit_hparams import AlphaEditHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: AlphaEditHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # # Inserts new "delta" variable at the appropriate part of the computation
    # def edit_output_fn(cur_out, cur_layer):
    #     nonlocal target_init

    #     if cur_layer == hparams.layer_module_tmp.format(layer):
    #         # Store initial value of the vector of interest
    #         if target_init is None:
    #             print("Recording initial value of v*")
    #             # Initial value is recorded for the clean sentence
    #             target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

    #         # Add intervened delta
    #         for i, idx in enumerate(lookup_idxs):

    #             if len(lookup_idxs)!=len(cur_out[0]):
    #                 cur_out[0][idx, i, :] += delta
    #             else:
    #                 cur_out[0][i, idx, :] += delta

    #     return cur_out
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Resolve the actual tensor (some backends wrap outputs in a tuple)
            t = cur_out[0] if isinstance(cur_out, (tuple, list)) else cur_out

            # Record initial v* once, from the clean sentence
            if target_init is None:
                print("Recording initial value of v*")
                # Expecting lookup_idxs[0] to be a sequence position
                if t.dim() == 3:
                    # (B, S, H): take batch 0 at that position
                    target_init = t[0, lookup_idxs[0]].detach().clone()
                elif t.dim() == 2:
                    # (S, H): take that position directly
                    target_init = t[lookup_idxs[0]].detach().clone()
                else:
                    raise RuntimeError(f"Unexpected activation rank: {t.dim()}")

            # Add intervened delta(s)
            for i, idx in enumerate(lookup_idxs):
                # ensure dtype/device match
                d = delta.to(t.dtype).to(t.device)

                if t.dim() == 3:
                    # (B, S, H)
                    B, S, H = t.shape

                    if B == len(lookup_idxs):
                        # Case A: one lookup index per batch item
                        #   i -> batch index, idx -> position
                        b = i
                        pos = idx
                    else:
                        # Case B: single (or few) batch item(s) with multiple positions
                        #   apply edit to batch 0 at each position
                        b = 0
                        pos = idx

                    if not (0 <= b < B and 0 <= pos < S):
                        raise IndexError(f"AlphaEdit indices out of range: b={b}/{B}, pos={pos}/{S}")
                    t[b, pos, :] += d

                elif t.dim() == 2:
                    # (S, H): batch got squeezed when B==1
                    S, H = t.shape
                    pos = idx
                    if not (0 <= pos < S):
                        raise IndexError(f"AlphaEdit position out of range: pos={pos}/{S}")
                    t[pos, :] += d

                else:
                    raise RuntimeError(f"Unexpected activation rank: {t.dim()} for AlphaEdit hook")

            # Return in the same container type that was passed in
            return (t,) if isinstance(cur_out, (tuple, list)) else t

        return cur_out


    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # # Compute loss on rewriting targets
        # output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        # if output.shape[1]!=rewriting_targets.shape[1]:
        #     output=torch.transpose(output, 0, 1)
        # full_repr = output[:len(rewriting_prompts)]

        # # log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        # # Normalize shapes
        # hs = ln_f(full_repr)  # same shape as full_repr, typically (B, S, H) or (S, H)

        # # Ensure we have (B, S, H)
        # if hs.dim() == 2:   # (S, H)
        #     hs = hs.unsqueeze(0)  # -> (1, S, H)
        # elif hs.dim() != 3:
        #     raise RuntimeError(f"Unexpected hidden-state rank: {hs.dim()}")

        # # Get lm head weights/bias safely
        # lm_head = getattr(model, hparams.lm_head_module)  # e.g., "lm_head"
        # lm_w = lm_head.weight      # usually shape (vocab, H)
        # lm_b = getattr(lm_head, "bias", None)

        # # Use F.linear, which handles (B, S, H) @ W^T correctly regardless of orientation
        # logits = torch.nn.functional.linear(
        #     hs,                                    # (B, S, H)
        #     lm_w.to(hs.dtype).to(hs.device),       # (vocab, H)
        #     lm_b.to(hs.dtype).to(hs.device) if lm_b is not None else None,
        # )  # -> (B, S, vocab)

        # log_probs = torch.log_softmax(logits, dim=-1)
        # --- Normalize traced activations to (B, S, H) safely ---
        loss_key = hparams.layer_module_tmp.format(loss_layer)

        # Resolve the actual tensor from the trace
        t_out = tr[loss_key].output
        t_out = t_out[0] if isinstance(t_out, (tuple, list)) else t_out

        # Determine the model hidden size
        hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd")

        if t_out.dim() == 3:
            # Candidates: (B, S, H), (S, B, H), (B, H, S), (S, H, B), ...
            if t_out.shape[-1] == hidden_size:
                rep = t_out  # (B, S, H) or (S, B, H) -> still need to check (B,S,?)
            elif t_out.shape[0] == hidden_size:
                rep = t_out.permute(1, 2, 0)  # (S, H, B) -> (S, B, H)
            elif t_out.shape[1] == hidden_size:
                rep = t_out.permute(0, 2, 1)  # (B, H, S) -> (B, S, H)
            else:
                raise RuntimeError(f"Cannot locate hidden_size={hidden_size} in 3D tensor with shape {t_out.shape}")

            # Now ensure it's (B, S, H): hidden must be last, seq middle
            if rep.shape[-1] != hidden_size:
                raise RuntimeError(f"Hidden not last after permute: {rep.shape}")
            # Heuristic: compare against rewriting_targets to decide which dim is S
            seq_len = rewriting_targets.shape[1]
            if rep.shape[1] != seq_len and rep.shape[0] == seq_len:
                rep = rep.permute(1, 0, 2)  # swap B and S

        elif t_out.dim() == 2:
            # Candidates: (S, H) or (H, S). Make (1, S, H).
            if t_out.shape[-1] == hidden_size:
                rep = t_out.unsqueeze(0)  # (1, S, H)
            elif t_out.shape[0] == hidden_size:
                rep = t_out.transpose(0, 1).unsqueeze(0)  # (1, S, H)
            else:
                raise RuntimeError(f"Cannot locate hidden_size={hidden_size} in 2D tensor with shape {t_out.shape}")
        else:
            raise RuntimeError(f"Unexpected activation rank: {t_out.dim()} with shape {t_out.shape}")

        # Batch-first slice for rewriting prompts
        full_repr = rep[:len(rewriting_prompts)]  # (B_rewrite, S, H)

        # --- Project to logits with correct shapes ---
        hs = ln_f(full_repr)  # (B_rewrite, S, H)

        lm_head = getattr(model, hparams.lm_head_module)  # e.g., "lm_head"
        lm_w = lm_head.weight
        lm_b = getattr(lm_head, "bias", None)

        logits = torch.nn.functional.linear(
            hs,                                    # (B, S, H)
            lm_w.to(hs.dtype).to(hs.device),       # (vocab, H)
            lm_b.to(hs.dtype).to(hs.device) if lm_b is not None else None,
        )  # (B, S, vocab)

        log_probs = torch.log_softmax(logits, dim=-1)


        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
