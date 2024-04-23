def format_docs(docs):
    return "\n\n".join(
        f"Doc path: {doc.metadata['source']} \n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )


def format_docs_with_index(docs):
    return "\n\n".join(
        f"Index: {i} \nDoc path: {doc.metadata['source']} \n{doc.page_content}"
        for i, doc in enumerate(docs, 1)
    )
