from .client import EmbedClient
from .adapter import LangChainEmbeddingsAdapter


def get_embed_client(
    model_name: str = "simple-hash", dim: int = 64, normalize: bool = True
) -> EmbedClient:
    """
    Create an EmbedClient with common defaults.
    """
    return EmbedClient(model_name=model_name, dim=dim, normalize=normalize)


def get_langchain_embeddings_adapter(
    embed_client, batch_size: int = 32
) -> LangChainEmbeddingsAdapter:
    return LangChainEmbeddingsAdapter(embed_client, batch_size=batch_size)
