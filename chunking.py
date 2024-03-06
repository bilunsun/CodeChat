from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
# import logging
# import sys
#
# logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def chunk_directory(directory: str, required_exts: list[str] = []) -> list[str]:
    docs = SimpleDirectoryReader(
        input_dir=directory,
        required_exts=required_exts,
        recursive=True,
    ).load_data()

    # Defaults
    text_parser = SentenceSplitter(
        chunk_size=256,
        chunk_overlap=64,
    )

    text_chunks = []
    for doc in docs:
        cur_text_chunks = text_parser.split_text(doc.text)
        text_chunks.extend(cur_text_chunks)

    # for chunk in text_chunks:
    #     print("-" * 80)
    #     print(chunk)
    # breakpoint()

    return text_chunks
