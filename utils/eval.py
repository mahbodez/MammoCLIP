import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from custom.mammodata import MammogramDataset
from custom.model import MammoCLIP
from tqdm.auto import tqdm


@torch.inference_mode()
def evaluate_birads(
    model: MammoCLIP,
    dataset: MammogramDataset,
    batch_size: int = 16,
    device: str = None,
    birads_classes: list = list(range(7)),
):
    """
    Evaluate a CLIP-based model on mammogram birads classification.

    Args:
        model: VisionTextDualEncoderModel that scores text vs images.
        dataset: MammogramDataset with df containing 'left_birads' and 'right_birads'.
        batch_size: batch size for inference.
        device: computation device (e.g., 'cuda' or 'cpu').
        birads_classes: list of class labels (0..6).

    Returns:
        dict: metrics including accuracy, precision, recall, f1, and confusion matrix.
    """
    # set device
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    # prepare class text embeddings
    classes = [f"BI-RADS {c}" for c in birads_classes]
    tokenizer = dataset.tokenizer
    text_inputs = tokenizer(classes, **dataset.tokenizer_kwargs)
    for k, v in text_inputs.items():
        text_inputs[k] = v.to(device)

    # precompute ground-truth labels array
    df = dataset.df
    left = df["left_birads"].astype(int).to_numpy()
    right = df["right_birads"].astype(int).to_numpy()
    labels_all = np.maximum(left, right)

    # DataLoader for batching
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in tqdm(
            enumerate(loader), total=len(loader), desc="Evaluating"
        ):
            # move images to device
            imgs = batch["pixel_values"].to(device)
            # forward pass: text vs image similarity
            outputs = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs.get("attention_mask"),
                token_type_ids=text_inputs.get("token_type_ids"),
                pixel_values=imgs,
                return_dict=True,
            )
            logits = outputs["logits_per_image"]  # (batch, n_classes)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()

            # slice true labels for this batch
            start = batch_idx * batch_size
            labels = labels_all[start : start + preds.shape[0]]

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # compute metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=birads_classes)
    # make a confusion matrix for the 7 classes
    cm = confusion_matrix(all_labels, all_preds, labels=birads_classes)

    # return metrics
    torch.cuda.empty_cache()
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }
