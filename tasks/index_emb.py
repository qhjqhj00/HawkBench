from my_own_tools import *
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.retrieval import DenseRetriever, FaissIndex

class IndexEmbPipeline:
    def __init__(self, 
        retriever: DenseRetriever, 
        save_path:str=""):
        
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.retriever = retriever

    def __call__(self, chunks, _idx):
        self.retriever.add(chunks)
        self.retriever._index.save(os.path.join(self.save_path, f"{_idx}.bin"))
        save_json(chunks, os.path.join(self.save_path, f"{_idx}.json"))
        self.retriever.remove_all()


if __name__ == "__main__":
    contexts = load_json("data/HawkBench/all_context.json")
    retriever = DenseRetriever(encoder="BAAI/bge-m3", cache_dir="")
    pipeline = IndexEmbPipeline(retriever, "data/HawkBench/index/emb/bge-m3")
    from semantic_text_splitter import TextSplitter
    from tqdm import tqdm

    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 512)
    for i,context_id in enumerate(contexts):
        context = contexts[context_id]["context"]
        chunks = splitter.chunks(context)
        print(f"{i+1}/{len(contexts)}: ", len(chunks))
        pipeline(chunks, context_id)
