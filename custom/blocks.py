import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewEmbedding(nn.Module):
    def __init__(self, num_views=4, embedding_dim=768):
        """
        Positional embedding for different mammogram views.
        Args:
            num_views (int): Number of mammogram views. Defaults to 4.
            embedding_dim (int): Dimension of the embedding. Defaults to 768.
        """
        super(ViewEmbedding, self).__init__()
        self.view_embed = nn.Embedding(num_views, embedding_dim)

    def forward(self, visual_embeddings: torch.FloatTensor):
        # visual_embeddings: (batch, num_views, embedding_dim)
        batch_size, num_views, _ = visual_embeddings.size()
        view_indices = (
            torch.arange(num_views, device=visual_embeddings.device)
            .unsqueeze(0)
            .repeat(batch_size, 1)
        )
        pos_embeddings = self.view_embed(view_indices)
        return visual_embeddings + pos_embeddings


class AttentionFusion(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = None):
        """
        Attention-based fusion of multiple embeddings (e.g., mammogram views).

        Args:
            embedding_dim (int): Dimension of input embeddings.
            hidden_dim (int, optional): Hidden dimension for attention network.
                                        Defaults to embedding_dim // 2.
        """
        super(AttentionFusion, self).__init__()

        if hidden_dim is None:
            hidden_dim = embedding_dim // 2

        # Small attention network
        self.attention_net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, view_embeddings: torch.FloatTensor):
        """
        Forward pass for attention fusion.

        Args:
            view_embeddings (Tensor): Embeddings of shape (batch_size, num_views, embedding_dim)

        Returns:
            fused_embedding (Tensor): Fused embedding (batch_size, embedding_dim)
            attention_weights (Tensor): Attention weights (batch_size, num_views)
        """
        # batch_size, num_views, embedding_dim = view_embeddings.shape

        # Compute attention scores
        attn_scores = self.attention_net(view_embeddings)  # (batch_size, num_views, 1)

        # Normalize scores to weights
        attn_weights = F.softmax(attn_scores, dim=1)  # (batch_size, num_views, 1)

        # Weighted sum of embeddings
        fused_embedding = torch.sum(
            attn_weights * view_embeddings, dim=1
        )  # (batch_size, embedding_dim)

        # Return fused embedding and attention weights (squeezed)
        return fused_embedding, attn_weights.squeeze(-1)
