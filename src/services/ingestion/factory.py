from .client import IngestionClient

def get_ingestion_client(source_type: str = "local", **kwargs) -> IngestionClient:
    if source_type == "local":
        return IngestionClient(**kwargs)
    raise ValueError(f"Unknown source_type: {source_type}")