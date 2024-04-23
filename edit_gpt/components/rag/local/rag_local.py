import re
import uuid
from operator import attrgetter
from typing import List, Optional

from langchain_community.vectorstores.faiss import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore

from edit_gpt.components.rag.base_rag import BaseRAGManager
from edit_gpt.components.rag.local.rag_prompts import FILTER_PROMPT
from edit_gpt.utils.loaders import load_docs_from_paths
from edit_gpt.utils.utils import format_docs_with_index


def extract_numbers(s):
    return [int(num) for num in re.findall(r"\d+", s)]


class LocalRAGManager(BaseRAGManager):
    def __init__(
        self,
        embeddings: Embeddings,
        paths_to_rag: Optional[List[str]] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 0,
        search_type: str = "mmr",
        search_kwargs: Optional[dict] = None,
        vectorstore: VectorStore = FAISS,
        chat_model: Optional[BaseChatModel] = None,
    ):
        self.paths_to_rag = paths_to_rag
        self.vectorstore = vectorstore
        self.chat_model = chat_model
        self.docs_indexes = set()
        super().__init__(
            embeddings, chunk_size, chunk_overlap, search_type, search_kwargs
        )

    def get_filtered_docs(self, question, **kwargs):
        filter_chain = (
            FILTER_PROMPT | self.chat_model | attrgetter("content") | extract_numbers
        )
        rag_retriever_input = question
        rag_retriever_input += "\n"
        rag_retriever_input += "\n".join([*kwargs])
        docs = self.rag_retriever.invoke(rag_retriever_input)

        filtered_docs_indexes = filter_chain.invoke(
            {
                "question": question,
                "rag_context": format_docs_with_index(docs),
                **kwargs,
            }
        )
        try:
            return [docs[i - 1] for i in filtered_docs_indexes]
        except IndexError:
            return docs

    def init_database(self):
        if self.paths_to_rag is not None:
            rag_docs = load_docs_from_paths(self.paths_to_rag)
            rag_docs = self.text_splitter.split_documents(rag_docs)
            ids = [str(uuid.uuid4()) for _ in rag_docs]
            for i, doc in enumerate(rag_docs):
                doc.metadata["id"] = ids[i]
            self.docs_indexes.update(ids)
            return self.vectorstore.from_documents(rag_docs, self.embeddings, ids=ids)
        else:
            # trick to init empty db
            fake_id = str(uuid.uuid4())
            db = self.vectorstore.from_documents(
                [Document(page_content="")], self.embeddings, ids=[fake_id]
            )
            db.delete([fake_id])
            return db

    def get_retriever(self):
        return self.rag_database.as_retriever(
            search_type=self.search_type,
            search_kwargs=self.search_kwargs,
        )

    def get_all_docs(self):
        db_length = len(self.docs_indexes)
        if db_length == 0:
            return []
        return self.rag_database.search("", "similarity", k=db_length)

    def add_texts_from_paths(self, paths):
        docs = load_docs_from_paths(paths)
        docs = self.text_splitter.split_documents(docs)
        ids = [str(uuid.uuid4()) for _ in docs]
        for i, doc in enumerate(docs):
            doc.metadata["id"] = ids[i]
        self.rag_database.add_documents(docs, ids=ids)
        self.docs_indexes.update(ids)

    def delete(self, doc_id: str) -> None:
        self.rag_database.delete([doc_id])
        self.docs_indexes.remove(doc_id)
