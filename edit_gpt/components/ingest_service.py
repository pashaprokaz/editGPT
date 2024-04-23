import os

from edit_gpt.components.rag.local.rag_local import LocalRAGManager
from edit_gpt.utils.loaders import get_all_filenames_from_paths


class IngestService:
    def __init__(self, rag_manager: LocalRAGManager):
        self._rag_manager = rag_manager
        self.ingested_files_last_edit_time = {}

    def list_ingested(self) -> list:
        return self._rag_manager.get_all_docs()

    def bulk_ingest(self, paths: list[str]) -> None:
        filenames = get_all_filenames_from_paths(paths)
        for path in filenames:
            timestamp = os.path.getmtime(path)
            self.ingested_files_last_edit_time[path] = timestamp
        self._rag_manager.add_texts_from_paths(paths)

    def delete(self, doc_id: str) -> None:
        self._rag_manager.delete(doc_id)
