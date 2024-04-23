import os

from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders.text import TextLoader


def load_doc(path):
    if path.endswith(".ipynb"):
        loader = NotebookLoader(
            path, include_outputs=True, max_output_length=20, remove_newline=True
        )
    else:
        loader = TextLoader(path, encoding="utf-8")

    try:
        docs = loader.load()
        return docs

    except RuntimeError:
        return None


def normalize_to_straight_slash(path):
    return os.path.normpath(path).replace("\\", "/")


def get_all_filenames_from_paths(paths):
    all_files = []
    for path in paths:
        if os.path.isfile(path):
            all_files.append(normalize_to_straight_slash(path))
        else:
            for root, dirs, files in os.walk(path):
                for file in files:
                    all_files.append(os.path.join(root, file))
                    all_files[-1] = normalize_to_straight_slash(all_files[-1])
    return all_files


def load_docs_from_paths(paths):
    all_files = get_all_filenames_from_paths(paths)
    docs = []
    for file_path in all_files:
        file_docs = load_doc(file_path)
        if file_docs is not None:
            docs += file_docs
    return docs
