from retrieval import DenseRetriever, FaissIndex
from flask import Flask, request, jsonify
import os
from semantic_text_splitter import TextSplitter
text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 512)
app = Flask(__name__)
import json
from utils import get_md5

retriever = DenseRetriever(encoder="BAAI/bge-m3", cache_dir="")

@app.route("/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query")
    hits = data.get("hits", 10)
    context = data.get("context", None)
    context_id = data.get("context_id", None)

    if context_id is not None and os.path.exists(f"data/index/{context_id}.index"):
        print("load existing index")
        index = FaissIndex(device="cuda:0")
        index.load(f"data/index/{context_id}.index")
        chunks = json.load(open(f"data/chunks/{context_id}.json"))
        retriever._index = index
        
    elif context is not None:
        print("add new index")
        chunks = text_splitter.chunks(context)
        context_id = get_md5(context[:1024]+context[-1024:])
        retriever.remove_all()
        retriever.add(chunks)
        #save chunks to file
        with open(f"data/chunks/{context_id}.json", "w") as f:
            json.dump(chunks, f, ensure_ascii=False)
        retriever._index.save(f"data/index/{context_id}.index")
        print("save new index")
    else:
        raise ValueError("context or context_id is required")
    
    scores, indices = retriever.search(query, hits)
    indices = indices.tolist()[0]
    print(indices)
    retrieved_chunks = [chunks[i] for i in indices]

    return jsonify({"indices": indices, "chunks": retrieved_chunks})  

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=39112, debug=False)  