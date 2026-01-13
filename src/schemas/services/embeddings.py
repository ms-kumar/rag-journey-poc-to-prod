"""Embedding service schemas."""

from pydantic import BaseModel, ConfigDict, Field


class SparseEncoderConfig(BaseModel):
    """Configuration for SPLADE encoder."""

    model_name: str = Field(
        "naver/splade-cocondenser-ensembledistil", description="HuggingFace model name for SPLADE"
    )
    device: str = Field("cpu", description="Device to run model on: cpu or cuda")
    batch_size: int = Field(32, description="Batch size for encoding", ge=1)
    max_length: int = Field(256, description="Maximum sequence length", ge=1)
    revision: str | None = Field(None, description="Model revision for reproducibility")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "naver/splade-cocondenser-ensembledistil",
                "device": "cpu",
                "batch_size": 32,
                "max_length": 256,
            }
        }
    )


class EmbeddingRequest(BaseModel):
    """Request for generating embeddings."""

    texts: list[str] = Field(..., description="Texts to embed", min_length=1)
    model: str | None = Field(None, description="Model to use for embeddings")
    batch_size: int | None = Field(None, description="Batch size override", ge=1)


class EmbeddingResponse(BaseModel):
    """Response with generated embeddings."""

    embeddings: list[list[float]] = Field(..., description="Generated embeddings")
    model: str = Field(..., description="Model used")
    usage: dict[str, int] | None = Field(None, description="Token usage information")


class SparseEmbeddingResponse(BaseModel):
    """Response with sparse embeddings."""

    sparse_embeddings: list[dict[int, float]] = Field(
        ..., description="Sparse embeddings as dict of token_id -> weight"
    )
    model: str = Field(..., description="Model used")
