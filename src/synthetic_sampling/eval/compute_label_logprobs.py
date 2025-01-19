import math
from tqdm import tqdm
from typing import List, Dict, Any
import torch

def compute_label_logprobs_batch_mc(
    model: torch.nn.Module,
    tokenizer,
    data_list: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 4  
) -> List[Dict[str, float]]:
    """
    Implements multiple-choice log-prob computation using batched model inference.

    Args:
        model: The language model.
        tokenizer: The tokenizer.
        data_list: List of dicts, each with:
            - 'prompt': str
            - 'label_options': List[str]
            - 'true_label': str
        device: torch.device
        batch_size: Number of (prompt + label) pairs to process per forward pass.

    Returns:
        List of dicts, each mapping label_option to probability for each data item.
    """
    model.eval()
    all_distributions = []
    N = len(data_list)

    # 1. Build all_pairs as list of tuples: (data_item_index, label)
    all_pairs = []
    for idx, item in enumerate(data_list):
        for label in item["label_options"]:
            all_pairs.append((idx, label))

    total_pairs = len(all_pairs)

    # 2. Placeholder to store sum_logprobs per data_item_index
    sum_logprobs_per_item = [[] for _ in range(N)]

    # 3. Process all_pairs in batches
    with tqdm(total=total_pairs, desc="Computing label logprobs", unit="pair") as pbar:
        for start in range(0, total_pairs, batch_size):
            end = min(start + batch_size, total_pairs)
            batch_slice = all_pairs[start:end]
            batch_labels = [pair[1] for pair in batch_slice]

            # Tokenize (prompt + label)
            combined_texts = [data_list[idx]["prompt"] + label for idx, label in batch_slice]
            tokenized = tokenizer(
                combined_texts,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(device)

            # Build labels tensor: mask prompt tokens
            labels = tokenized['input_ids'].clone()
            for i, (idx, label) in enumerate(batch_slice):
                prompt = data_list[idx]["prompt"]
                prompt_len = len(tokenizer.encode(prompt, add_special_tokens=False))
                labels[i, :prompt_len] = -100  # Mask prompt tokens

            with torch.no_grad():
                outputs = model(
                    input_ids=tokenized['input_ids'],
                    attention_mask=tokenized['attention_mask'],
                    labels=labels
                )
                
                label_token_counts = [len(tokenizer.encode(label, add_special_tokens=False)) for label in batch_labels]
                label_token_counts_tensor = torch.tensor(label_token_counts, device=device, dtype=torch.float)
                sum_logprobs_tensor = outputs.loss * label_token_counts_tensor
                sum_logprobs_tensor = -sum_logprobs_tensor
                sum_logprobs = sum_logprobs_tensor.cpu().numpy().tolist()

            for i, (idx, label) in enumerate(batch_slice):
                sum_lp = sum_logprobs[i]
                sum_logprobs_per_item[idx].append((label, sum_lp))

            pbar.update(len(batch_slice))

    # 4. for each data item, compute probabilities via softmax
    for item_idx in range(N):
        label_logprobs = sum_logprobs_per_item[item_idx]  # list of (label, sum_logprob)
        labels = [ll[0] for ll in label_logprobs]
        logprobs = [ll[1] for ll in label_logprobs]

        max_lp = max(logprobs)
        exps = [math.exp(lp - max_lp) for lp in logprobs]
        denom = sum(exps)
        probs = [e / denom for e in exps]

        dist_dict = {label: prob for label, prob in zip(labels, probs)}
        all_distributions.append(dist_dict)

    return all_distributions

