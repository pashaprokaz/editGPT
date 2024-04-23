from abc import ABC, abstractmethod

from langchain_text_splitters import CharacterTextSplitter


class BaseRAGManager(ABC):
    def __init__(
        self,
        embeddings,
        chunk_size=1000,
        chunk_overlap=0,
        search_type="mmr",
        search_kwargs=None,
    ):
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.embeddings = embeddings
        self.text_splitter = CharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        self.rag_database = self.init_database()
        self.rag_retriever = self.get_retriever()

    @abstractmethod
    def init_database(self):
        pass

    @abstractmethod
    def get_retriever(self):
        pass

    @abstractmethod
    def get_all_docs(self):
        pass

    @abstractmethod
    def add_texts_from_paths(self, paths):
        pass
