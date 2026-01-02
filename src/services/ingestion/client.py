from typing import List
from pathlib import Path


class IngestionClient:
    def __init__(self, directory: str = "./data"):
        self.directory = Path(directory)

    def ingest(self) -> List[str]:
        """
        Reads all .txt files from the directory and returns their contents as a list of strings.
        """
        documents = []
        for file_path in self.directory.glob("*.txt"):
            with file_path.open("r", encoding="utf-8") as f:
                documents.append(f.read())
        return documents
