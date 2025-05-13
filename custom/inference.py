# filepath: /Users/mahbod/Documents/VSCode/MammoCLIP/custom/inference.py
import torch
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# compute auc
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer
from custom.model import MammoCLIP
from custom.config import Config
from custom.preprocessing import MammogramPreprocessor, MammogramTransform
import cv2


def load_model(model_dir: str, config: Config, device: str = None) -> MammoCLIP:
    """
    Load a pretrained MammoCLIP model from the given directory and move to device.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = MammoCLIP.from_pretrained(model_dir, **config.pretrained_model_cfg)
    model.eval().to(device)
    return model


@torch.inference_mode()
def embed_queries(
    queries: list[str],
    model: MammoCLIP,
    config: Config,
    tokenizer: AutoTokenizer = None,
    device: str = None,
) -> torch.Tensor:
    """
    Tokenizes and encodes a list of text queries into normalized embeddings using the provided MammoCLIP model.

    Args:
        queries (list[str]): List of text queries to embed.
        model (MammoCLIP): The MammoCLIP model used to generate text embeddings.
        config (Config): Configuration object containing tokenizer and dataset settings.
        tokenizer (AutoTokenizer, optional): Tokenizer to use for encoding the queries. If None, a tokenizer is loaded from the model's text model path.
        device (str, optional): Device to perform computation on. If None, uses the device of the model's parameters.

    Returns:
        torch.Tensor: A tensor of normalized text embeddings with shape (len(queries), embedding_dim).
    """
    if device is None:
        device = model.device
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model.text_model.name_or_path)
    inputs = tokenizer(queries, **config.val_ds["attrs_"]["tokenizer_kwargs"])
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    text_emb = model.get_text_features(**inputs)
    text_emb /= text_emb.norm(dim=-1, keepdim=True)
    return text_emb


@torch.inference_mode()
def embed_views(
    views: list[str],
    model: MammoCLIP,
    config: Config,
    preprocessor: MammogramPreprocessor = None,
    transform: MammogramTransform = None,
    device: str = None,
) -> torch.Tensor:
    """
    Embeds a set of mammogram views into a normalized feature vector using the MammoCLIP model.

    Args:
        views (list[str]): List of file paths to mammogram images (one per view).
        model (MammoCLIP): Pretrained MammoCLIP model for feature extraction.
        config (Config): Configuration object containing preprocessing and transform settings.
        preprocessor (MammogramPreprocessor, optional): Preprocessing function or object. If None, loaded from config.
        transform (MammogramTransform, optional): Transformation function or object. If None, loaded from config.
        device (str, optional): Device to run the model on. If None, inferred from model parameters.

    Returns:
        torch.Tensor: Normalized embedding tensor representing the input mammogram views.
    """
    if device is None:
        device = model.device
    if preprocessor is None:
        preprocessor = MammogramPreprocessor.from_dict(
            config.val_ds["attrs_"]["image_preprocessor"]["attrs_"]
        )
    if transform is None:
        transform = MammogramTransform.from_dict(
            config.val_ds["attrs_"]["transform_function"]["attrs_"]
        )
    # Take a look at how the images are loaded and preprocessed in #MammoDataset's __getitem__
    imgs = [cv2.imread(view, cv2.IMREAD_GRAYSCALE).astype(np.float32) for view in views]
    processed_imgs = [preprocessor(img) for img in imgs]
    processed_imgs = np.concatenate(processed_imgs, axis=0)  # (num_views, H, W)

    processed_imgs = [transform(img) for img in processed_imgs]
    processed_imgs = torch.stack(processed_imgs, dim=0).squeeze()  # (num_views, H, W)

    tensor = processed_imgs.unsqueeze(0).to(device)  # (1, num_views, H, W)
    img_emb = model.get_image_features(pixel_values=tensor)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)
    return img_emb


@torch.inference_mode()
def batch_embed_views(
    view_batch: list[list[str]],
    model: MammoCLIP,
    config: Config,
    preprocessor: MammogramPreprocessor = None,
    transform: MammogramTransform = None,
    device: str = None,
) -> torch.Tensor:
    """
    Embeds a batch of mammogram views into normalized feature vectors using the MammoCLIP model.

    Args:
        views (list[list[str]]): List of lists, where each inner list contains file paths to mammogram images (one per view).
        model (MammoCLIP): Pretrained MammoCLIP model for feature extraction.
        config (Config): Configuration object containing preprocessing and transform settings.
        preprocessor (MammogramPreprocessor, optional): Preprocessing function or object. If None, loaded from config.
        transform (MammogramTransform, optional): Transformation function or object. If None, loaded from config.
        device (str, optional): Device to run the model on. If None, inferred from model parameters.
    Returns:
        torch.Tensor: Normalized embedding tensor representing the input mammogram views.
    """
    if device is None:
        device = model.device
    if preprocessor is None:
        preprocessor = MammogramPreprocessor.from_dict(
            config.val_ds["attrs_"]["image_preprocessor"]["attrs_"]
        )
    if transform is None:
        transform = MammogramTransform.from_dict(
            config.val_ds["attrs_"]["transform_function"]["attrs_"]
        )
    # Take a look at how the images are loaded and preprocessed in #MammoDataset's __getitem__
    imgs = [
        [cv2.imread(view, cv2.IMREAD_GRAYSCALE).astype(np.float32) for view in views]
        for views in view_batch
    ]
    processed_imgs = [[preprocessor(img) for img in views] for views in imgs]
    processed_imgs = [
        np.concatenate(views, axis=0) for views in processed_imgs
    ]  # (num_views, H, W) * batch
    processed_imgs = [[transform(img) for img in views] for views in processed_imgs]
    processed_imgs = [
        torch.stack(views, dim=0).squeeze() for views in processed_imgs
    ]  # (num_views, H, W) * batch
    tensor = torch.stack(processed_imgs, dim=0).to(
        device
    )  # (batch_size, num_views, H, W)
    img_emb = model.get_image_features(pixel_values=tensor)
    img_emb /= img_emb.norm(dim=-1, keepdim=True)
    return img_emb


def infer_single_mammogram(
    views: list[str],
    queries: list[str],
    model: MammoCLIP,
    config: Config,
    preprocessor: MammogramPreprocessor = None,
    transform: MammogramTransform = None,
    tokenizer: AutoTokenizer = None,
    device: str = None,
) -> np.ndarray:
    """
    Performs inference on a single mammogram using provided image views and text queries.

    Args:
        views (list[str]): List of file paths or identifiers for mammogram image views.
        queries (list[str]): List of textual queries or labels to evaluate.
        model (MammoCLIP): The MammoCLIP model instance for embedding and inference.
        config (Config): Configuration object containing model and preprocessing parameters.
        preprocessor (MammogramPreprocessor, optional): Preprocessing function or object for mammogram images. Defaults to None.
        transform (MammogramTransform, optional): Additional image transformation to apply. Defaults to None.
        tokenizer (AutoTokenizer, optional): Tokenizer for processing text queries. Defaults to None.
        device (str, optional): Device identifier (e.g., 'cpu' or 'cuda') for computation. Defaults to None.

    Returns:
        np.ndarray: Array of probability scores (shape: (num_queries,)) corresponding to each query.
    """
    views_emb = embed_views(views, model, config, preprocessor, transform, device)
    # view embeddings are of (1, dim) shape
    queries_emb = embed_queries(queries, model, config, tokenizer, device)
    # queries_emb is of (num_queries, dim) shape
    logits = 100.0 * views_emb @ queries_emb.T
    # logits is of (1, num_queries) shape
    probs = torch.softmax(logits, dim=-1)
    # probs is of (1, num_queries) shape
    return probs.cpu().squeeze().numpy()  # (num_queries,)


@torch.inference_mode()
def infer_from_views(
    views: list[str],
    query_embeddings: torch.Tensor,
    model: MammoCLIP,
    config: Config,
    preprocessor: MammogramPreprocessor = None,
    transform: MammogramTransform = None,
    tau: float = None,
    device: str = None,
) -> np.ndarray:
    """
    Computes the similarity probabilities between a set of mammogram views and query embeddings using a MammoCLIP model.

    Args:
        views (list[str]): List of file paths or identifiers for mammogram views to be embedded.
        query_embeddings (torch.Tensor): Tensor of shape (num_queries, embedding_dim) containing query embeddings to compare against.
        model (MammoCLIP): The MammoCLIP model used to generate embeddings for the views.
        config (Config): Configuration object for the model and preprocessing.
        preprocessor (MammogramPreprocessor, optional): Preprocessing function or object to apply to each view before embedding.
        transform (MammogramTransform, optional): Additional transformation to apply to each view.
        tau (float, optional): Temperature scaling factor for logits. If None, uses the model's logit scale.
        device (str, optional): Device to perform computation on (e.g., 'cpu' or 'cuda'). If None, uses default device.

    Returns:
        np.ndarray: Array of probabilities (shape: (num_queries,)) representing the softmax-normalized similarity between the embedded views and each query embedding.
    """
    if tau is None:
        tau = model.logit_scale.exp().item()
    views_emb = embed_views(views, model, config, preprocessor, transform, device)
    # view embeddings are of (1, dim) shape
    logits = tau * views_emb @ query_embeddings.T
    # logits is of (1, num_queries) shape
    probs = torch.softmax(logits, dim=-1)
    # probs is of (1, num_queries) shape
    return probs.cpu().squeeze().numpy()  # (num_queries,)


@torch.inference_mode()
def infer_from_view_batch(
    view_batch: list[list[str]],
    query_embeddings: torch.Tensor,
    model: MammoCLIP,
    config: Config,
    preprocessor: MammogramPreprocessor = None,
    transform: MammogramTransform = None,
    tau: float = None,
    device: str = None,
) -> np.ndarray:
    """
    Computes the similarity probabilities between a batch of mammogram views and query embeddings using a MammoCLIP model.

    Args:
        view_batch (list[list[str]]): List of lists, where each inner list contains file paths or identifiers for mammogram views to be embedded.
        query_embeddings (torch.Tensor): Tensor of shape (num_queries, embedding_dim) containing query embeddings to compare against.
        model (MammoCLIP): The MammoCLIP model used to generate embeddings for the views.
        config (Config): Configuration object for the model and preprocessing.
        preprocessor (MammogramPreprocessor, optional): Preprocessing function or object to apply to each view before embedding.
        transform (MammogramTransform, optional): Additional transformation to apply to each view.
        tau (float, optional): Temperature scaling factor for logits. If None, uses the model's logit scale.
        device (str, optional): Device to perform computation on (e.g., 'cpu' or 'cuda'). If None, uses default device.

    Returns:
        np.ndarray: Array of probabilities (shape: (batch_size, num_queries)) representing the softmax-normalized similarity between the embedded views and each query embedding.
    """
    if tau is None:
        tau = model.logit_scale.exp().item()
    views_emb = batch_embed_views(
        view_batch, model, config, preprocessor, transform, device
    )
    # view embeddings are of (batch_size, dim) shape
    logits = tau * views_emb @ query_embeddings.T
    # logits is of (batch_size, num_queries) shape
    probs = torch.softmax(logits, dim=-1)
    # probs is of (batch_size, num_queries) shape
    return probs.cpu().numpy()  # (batch_size, num_queries)


def make_views_batches(
    dataframe: pd.DataFrame,
    view_cols: list[str],
    batch_size: int = 4,
) -> list[list[list[str]]]:
    """
    Creates a batch of views from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data.
        view_cols (list[str]): List of column names in the DataFrame corresponding to different mammogram views.
        batch_size (int, optional): The size of the batch to create. Defaults to 4.

    Returns:
        list[list[list[str]]]: A list of batches, where each batch is a list of views.
    """
    batches = []
    for i in range(0, len(dataframe), batch_size):
        batch = []
        for j in range(batch_size):
            if i + j < len(dataframe):
                views = [dataframe.iloc[i + j][col] for col in view_cols]
                batch.append(views)
        batches.append(batch)
    return batches


@torch.inference_mode()
def evaluate(
    dataframe: pd.DataFrame,
    view_cols: list[str],
    query2label: dict,
    label_col: str,
    model: MammoCLIP,
    config: Config,
    device: str = None,
) -> dict:
    """
    Evaluates a MammoCLIP model on a given dataset and computes classification metrics.
    This function processes a DataFrame containing mammogram data, applies preprocessing and transformation,
    embeds the provided queries (labels), and performs inference using the MammoCLIP model. It then compares
    the predicted labels to the ground truth and computes various evaluation metrics.
    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data to evaluate.
        view_cols (list[str]): List of column names in the DataFrame corresponding to different mammogram views.
        label_col (str): Name of the column in the DataFrame containing the ground truth labels.
        queries (list[str]): List of label strings (queries) to be embedded and used for classification.
        model (MammoCLIP): The MammoCLIP model instance to use for inference.
        config (Config): Configuration object containing preprocessing and transformation settings.
        query2label (dict, optional): Dictionary mapping query strings to their corresponding labels.
            Useful for matching with queries. Defaults to None.
    Returns:
        dict: A dictionary containing the following keys:
            - "accuracy": (float) Overall classification accuracy.
            - "report": (str) Detailed classification report (precision, recall, f1-score per class).
            - "confusion_matrix": (np.ndarray) Confusion matrix of true vs. predicted labels.
            - "auc": (float) Macro-averaged ROC AUC score for multi-class classification.
            - "y_true": (np.ndarray) Array of true labels.
            - "y_pred": (np.ndarray) Array of predicted labels.
            - "y_scores": (np.ndarray) Array of predicted scores for each query.
            - "query2label": (dict) Mapping of query strings to their corresponding labels.
            - "label2query": (dict) Mapping of labels to their corresponding query strings.
            - "label2index": (dict) Mapping of labels to their corresponding indices.
    Raises:
        ValueError: If the number of unique labels in the label column does not match the number of queries,
            or if any label in the label column is not present in the queries list.
    Notes:
        - The function assumes that the queries list is ordered and matches the intended label indices.
        - The function uses tqdm for progress visualization during inference.
        - The function supports both CPU and CUDA devices for inference.
    """
    if device is None:
        device = model.device
    preprocessor = MammogramPreprocessor.from_dict(
        config.val_ds["attrs_"]["image_preprocessor"]["attrs_"]
    )
    transform = MammogramTransform.from_dict(
        config.val_ds["attrs_"]["transform_function"]["attrs_"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model.text_model.name_or_path)

    queries = list(query2label.keys())

    query_embeddings = embed_queries(queries, model, config, tokenizer, device)

    df = dataframe.copy(deep=True)
    label2query = {v: k for k, v in query2label.items()}
    label2index = {label: i for i, label in enumerate(query2label.values())}
    if not all(label2query[label] in queries for label in set(df[label_col])):
        raise ValueError(
            f"Some labels in {label_col} are not present in the queries list."
        )
    y_true = [label2index[label] for label in df[label_col].values]
    y_pred = []
    y_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        views = [row[c] for c in view_cols]
        scores = infer_from_views(
            views, query_embeddings, model, config, preprocessor, transform, device
        )
        pred_index = int(np.argmax(scores))
        y_pred.append(pred_index)
        y_scores.append(scores)
    # Convert y_pred and y_true to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(label2index.values()), target_names=queries
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(label2index.values()))
    if len(queries) == 2:  # binary classification
        auc = roc_auc_score(y_true, y_scores[:, 1], average="macro")
    else:  # multi-class classification
        auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "query2label": query2label,
        "label2query": label2query,
        "label2index": label2index,
    }


@torch.inference_mode()
def evaluate_batch(
    dataframe: pd.DataFrame,
    view_cols: list[str],
    query2label: dict,
    label_col: str,
    model: MammoCLIP,
    config: Config,
    batch_size: int = 4,
    device: str = None,
) -> dict:
    """
    Evaluates a MammoCLIP model on a given dataset and computes classification metrics.
    This function processes a DataFrame containing mammogram data, applies preprocessing and transformation,
    embeds the provided queries (labels), and performs inference using the MammoCLIP model. It then compares
    the predicted labels to the ground truth and computes various evaluation metrics.
    Args:
        dataframe (pd.DataFrame): The input DataFrame containing the data to evaluate.
        view_cols (list[str]): List of column names in the DataFrame corresponding to different mammogram views.
        label_col (str): Name of the column in the DataFrame containing the ground truth labels.
        queries (list[str]): List of label strings (queries) to be embedded and used for classification.
        model (MammoCLIP): The MammoCLIP model instance to use for inference.
        config (Config): Configuration object containing preprocessing and transformation settings.
        batch_size (int, optional): The size of the batch to create. Defaults to 4.
        query2label (dict, optional): Dictionary mapping query strings to their corresponding labels.
            Useful for matching with queries. Defaults to None.
    Returns:
        dict: A dictionary containing the following keys:
            - "accuracy": (float) Overall classification accuracy.
            - "report": (str) Detailed classification report (precision, recall, f1-score per class).
            - "confusion_matrix": (np.ndarray) Confusion matrix of true vs. predicted labels.
            - "auc": (float) Macro-averaged ROC AUC score for multi-class classification.
            - "y_true": (np.ndarray) Array of true labels.
            - "y_pred": (np.ndarray) Array of predicted labels.
            - "y_scores": (np.ndarray) Array of predicted scores for each query.
            - "query2label": (dict) Mapping of query strings to their corresponding labels.
            - "label2query": (dict) Mapping of labels to their corresponding query strings.
            - "label2index": (dict) Mapping of labels to their corresponding indices.
    Raises:
        ValueError: If the number of unique labels in the label column does not match the number of queries,
            or if any label in the label column is not present in the queries list.
    Notes:
        - The function assumes that the queries list is ordered and matches the intended label indices.
        - The function uses tqdm for progress visualization during inference.
        - The function supports both CPU and CUDA devices for inference.
    """
    if device is None:
        device = model.device
    preprocessor = MammogramPreprocessor.from_dict(
        config.val_ds["attrs_"]["image_preprocessor"]["attrs_"]
    )
    transform = MammogramTransform.from_dict(
        config.val_ds["attrs_"]["transform_function"]["attrs_"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model.text_model.name_or_path)

    queries = list(query2label.keys())

    query_embeddings = embed_queries(queries, model, config, tokenizer, device)

    df = dataframe.copy(deep=True)
    label2query = {v: k for k, v in query2label.items()}
    label2index = {label: i for i, label in enumerate(query2label.values())}
    if not all(label2query[label] in queries for label in set(df[label_col])):
        raise ValueError(
            f"Some labels in {label_col} are not present in the queries list."
        )
    y_true = [label2index[label] for label in df[label_col].values]
    y_pred = []
    y_scores = []
    batches = make_views_batches(df, view_cols, batch_size)
    for batch in tqdm(batches, desc=f"Evaluating {label_col}"):
        scores = infer_from_view_batch(
            batch,
            query_embeddings,
            model,
            config,
            preprocessor,
            transform,
            device,
        ).tolist()
        pred_index = np.argmax(scores, axis=-1).tolist()
        y_pred.extend(pred_index)
        y_scores.extend(scores)
    # Convert y_pred and y_true to numpy arrays
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(
        y_true, y_pred, labels=list(label2index.values()), target_names=queries
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(label2index.values()))
    if len(queries) == 2:
        auc = roc_auc_score(y_true, y_scores[:, 1], average="macro")
    else:
        auc = roc_auc_score(y_true, y_scores, multi_class="ovr", average="macro")
    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm,
        "auc": auc,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_scores": y_scores,
        "query2label": query2label,
        "label2query": label2query,
        "label2index": label2index,
    }
