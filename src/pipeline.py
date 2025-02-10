from .retrieval import DenseRetriever, FaissIndex
from .models import HuggingFaceModel, init_args
from my_own_tools import *
import os
from semantic_text_splitter import TextSplitter
from typing import Union, Dict
from tasks.prompts import _prompt_memoRAG, _instruct_sur, _instruct_span, _instruct_qa, hyde, query_rewrite
from itertools import chain
from transformers import QuantizedCacheConfig, QuantoQuantizedCache
from nano_graphrag import GraphRAG, QueryParam
from nano_graphrag._utils import wrap_embedding_func_with_attrs
from sentence_transformers import SentenceTransformer

EMBED_MODEL = SentenceTransformer(
    "BAAI/bge-m3", cache_folder="", device="cpu"
)
@wrap_embedding_func_with_attrs(
    embedding_dim=EMBED_MODEL.get_sentence_embedding_dimension(),
    max_token_size=EMBED_MODEL.max_seq_length,
)
async def local_embedding(texts: list[str]) -> np.ndarray:
    return EMBED_MODEL.encode(texts, normalize_embeddings=True)

def get_pre_cached_index(path, device):
    rtn = {}
    for file in os.listdir(path):
        if file.endswith(".json"):
            _id = file.split(".")[0]
            if os.path.exists(os.path.join(path, f"{_id}.bin")):
                _index = FaissIndex(device)
                _index.load(os.path.join(path,  f"{_id}.bin"))
            else:
                _index = None
            corpus = load_json(os.path.join(path, f"{_id}.json"))
            rtn[_id] = {"index": _index, "corpus": corpus}
    return rtn

def get_pipeline(model_args, device="cpu", **kwargs):
    model_kwargs, tokenizer_kwargs = init_args(
        model_args, 
        model_args.gen_model, device)
    
    model_args_dict = model_args.to_dict()
    pipeline_name = model_args_dict["pipeline"]
    generation_kwargs = {}
    if model_args_dict["gen_max_new_tokens"]:
        generation_kwargs["max_new_tokens"] = model_args_dict["gen_max_new_tokens"]
    if model_args_dict["gen_do_sample"]:
        generation_kwargs["do_sample"] = model_args_dict["gen_do_sample"]
    if model_args_dict["gen_temperature"]:
        generation_kwargs["temperature"] = model_args_dict["gen_temperature"]
    if model_args_dict["gen_top_p"]:
        generation_kwargs["top_p"] = model_args_dict["gen_top_p"]
    if model_args_dict["cache_implementation"]:
        generation_kwargs["cache_implementation"] = model_args_dict["cache_implementation"]
        generation_kwargs["cache_config"] ={
            "backend": model_args_dict["cache_backend"], 
            "nbits": model_args_dict["cache_nbits"]}
            
    if pipeline_name in ["longllm", "rag", "memorag", "quant", "hyde", "lingua", "graphrag"]:
        api_dict = load_json(model_args_dict["api_dict"])
        if model_args_dict["use_api"]:
            gen_model = Agent(model_args_dict["agent_model"], model_args_dict["agent_provider"], api_dict)
        
        else:
            gen_model_name = model_args_dict["gen_model"]
            gen_model = HuggingFaceModel(
                gen_model_name,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs
            )
            if model_args_dict["use_minference"]:
                from minference import MInference
                minference_patch = MInference("minference", "Qwen/Qwen2-7B-Instruct")
                gen_model.model = minference_patch(gen_model.model)

    if pipeline_name in ["rag", "memorag", "hyde"]:
        retriever  = DenseRetriever(
                encoder=model_args_dict["ret_model"],
                pooling_method=model_args_dict["ret_dense_pooling"],
                dense_metric=model_args_dict["ret_dense_metric"],
                query_max_length=model_args_dict["ret_dense_q_max_len"],
                key_max_length=model_args_dict["ret_dense_k_max_len"],
                cache_dir=model_args_dict["model_cache_dir"],
                hits=model_args_dict["ret_hits"],
            )
        if model_args.index_path:
            pre_cached_index = get_pre_cached_index(model_args.index_path, "cpu")
            print(f"{len(pre_cached_index)} indices loaded.")
        else:
            pre_cached_index = None

    if pipeline_name == "memorag":
        memory = HuggingFaceModel(
            model_args_dict["mem_model"],
            model_kwargs=model_kwargs,
            tokenizer_kwargs=tokenizer_kwargs,
        )


    if pipeline_name == "graphrag":

        pipeline = graphRAGPipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs
            # retriever=EMBED_MODEL
        )

    elif pipeline_name == "quant":
        pipeline = QuantLLMPipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs
        )
        
    elif pipeline_name == "rag":
        pipeline = RAGPipeline(
            retriever=retriever,
            generator=gen_model,
            gen_kwargs=generation_kwargs,
            pre_cached=pre_cached_index

        )
    elif pipeline_name == "lingua":
        from llmlingua import PromptCompressor
        compressor = PromptCompressor(
                model_name="llmlingua-2-xlm-roberta-large-meetingbank",
                use_llmlingua2=True,
                device=device
            )
        pipeline = LinguaMPipeline(
            generator=gen_model,
            prompt_compressor=compressor,
            generation_kwargs=generation_kwargs
        )
    elif pipeline_name == "longllm":
        pipeline = LongLLMPipeline(
            generator=gen_model,
            generation_kwargs=generation_kwargs)
    
    elif pipeline_name == "hyde":
        pipeline = HydePipeline(
            retriever=retriever,
            generator=gen_model,
            gen_kwargs=generation_kwargs,
            pre_cached=pre_cached_index

        )

    elif pipeline_name == "memorag":
        pipeline = MemoRAGPipeline(
            memory=memory,
            retriever=retriever,
            generator=gen_model,
            mem_gen_kwargs=generation_kwargs,
            pre_cached=pre_cached_index,
            gen_kwargs=generation_kwargs)
    else:
        raise NotImplementedError
            
    return pipeline

class QuantLLMPipeline:
    def __init__(self, generator: Union[HuggingFaceModel], generation_kwargs:Dict={}):
        self.generator = generator
        self.generation_kwargs = generation_kwargs

    def __call__(self, context:str, question:str, prompt:str, task="", cache_id="", conv=False):
        """
        Directly answer the question based on the context using the generator.
        """
        context_prompt = _prompt_memoRAG.format(context=context)
        inputs = self.generator.template2ids([{"role": "user", "content": context_prompt},
            {"role": "assistant", "content": "I have read the article. Please provide your question."}])

        cache_config = QuantizedCacheConfig(nbits=4)
        past_key_values = QuantoQuantizedCache(cache_config=cache_config)
        # past_key_values = SinkCache(window_length=4096, num_sink_tokens=128)

        outputs = self.generator.model(**inputs, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values
        
        if question:
            answer_prompt = _instruct_qa.format(question=question)
        else:
            answer_prompt = "summarize the long article."
        answer_output = self.generator.generate(answer_prompt, **self.generation_kwargs, past_key_values=past_key_values)
        return answer_output

class graphRAGPipeline:
    def __init__(self, generator, generation_kwargs: Dict={}, working_dir:str="data/index/graph"):

        # self.EMBED_MODEL = retriever
        self.working_dir = working_dir
        self.generator = generator
        self.generation_kwargs = generation_kwargs

    def __call__(self, context, question, prompt, cache_id="", task="", conv=False):
        rag = GraphRAG(
                working_dir=f"{self.working_dir}/{cache_id}",
                embedding_func=local_embedding,
            )
        knowledge = rag.query(question, param=QueryParam(mode="local", only_need_context=True))
        answer_prompt = prompt.format(input=question, context=knowledge)
        answer = self.generator.generate(answer_prompt, **self.generation_kwargs)
        return answer

class RAGPipeline:
    def __init__(self, 
        retriever: DenseRetriever, 
        generator: HuggingFaceModel, 
        gen_kwargs: Dict={},
        retrieval_chunk_size:int=1024,
        pre_cached=None):
        if pre_cached is not None:
            self.pre_cached = pre_cached
        else:
            self.pre_cached = {}

        self.retriever = retriever
        self.generator = generator
        self.generation_kwargs = gen_kwargs
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", retrieval_chunk_size)
    def __call__(self, context, question, prompt, cache_id="", task="",conv=False):

        if cache_id and cache_id in self.pre_cached:
            retrieval_corpus = self.pre_cached[cache_id]["corpus"]
            # print(f"use pre-cached: {cache_id}")
            index = self.pre_cached[cache_id]["index"]
            self.retriever._index = index
        else:
            cache_id = ""
            retrieval_corpus = self.text_splitter.chunks(context)
            self.retriever.add(retrieval_corpus)

        _, topk_indices = self.retriever.search(
            queries=question)
        topk_indices = sorted([x for x in topk_indices[0].tolist() if x > -1])

        retrieval_results = [retrieval_corpus[i].strip() for i in topk_indices]
        knowledge = "\n\n".join(retrieval_results)

        if question:
            answer_prompt = prompt.format(input=question, context=knowledge)
        else:
            answer_prompt = prompt.format(context=knowledge)
        # if not cache_id:   
        if cache_id not in self.pre_cached: 
            self.retriever.remove_all()
        if isinstance(self.generator, Agent):
            answer = self.generator.chat_completion(answer_prompt, max_completion_tokens=self.generation_kwargs["max_new_tokens"])
        else:
            answer = self.generator.generate(answer_prompt, **self.generation_kwargs)
        return answer
    

class LongLLMPipeline:
    def __init__(self, generator: Union[HuggingFaceModel], generation_kwargs:Dict={}):
        self.generator = generator
        self.generation_kwargs = generation_kwargs

    def __call__(self, context:str, question:str, prompt:str, task="", cache_id="", conv=False):
        """
        Directly answer the question based on the context using the generator.
        """
        if question:
            answer_prompt = prompt.format(input=question, context=context)
        else:
            answer_prompt = prompt.format(context=context)

        if isinstance(self.generator, Agent):
            answer = self.generator.chat_completion(answer_prompt, max_completion_tokens=self.generation_kwargs["max_new_tokens"])
        else:
            answer = self.generator.generate(answer_prompt, **self.generation_kwargs)
        return answer

class LinguaMPipeline:
    def __init__(self, generator: Union[HuggingFaceModel], prompt_compressor, generation_kwargs:Dict={}):
        self.generator = generator
        self.generation_kwargs = generation_kwargs
        self.compressor = prompt_compressor


    def __call__(self, context:str, question:str, prompt:str, task="", cache_id="", conv=False):
        """
        Directly answer the question based on the context using the generator.
        """
        if question:
            answer_prompt = prompt.format(input=question, context=context)
        else:
            answer_prompt = prompt.format(context=context)

        results = self.compressor.compress_prompt_llmlingua2(
            answer_prompt,
            rate=0.6,
            force_tokens=['\n', '.', '!', '?', ','],
            chunk_end_tokens=['.', '\n'],
            return_word_label=True,
            drop_consecutive=True
        )
        compressed_prompt = results['compressed_prompt']

        answer_output = self.generator.generate(compressed_prompt, **self.generation_kwargs)
        return answer_output

class MemoRAGPipeline:
    def __init__(self, 
    memory: Union[HuggingFaceModel], 
    retriever: Union[DenseRetriever], 
    generator: Union[HuggingFaceModel], 
    mem_gen_kwargs: Dict={},
    gen_kwargs: Dict={},
    retrieval_chunk_size:int=512,
    mem_type:str="all",
    pre_cached = None):
        print(memory.model_name_or_path)
        self.memory = memory
        self.retriever = retriever
        self.generator = generator

        self.retrieval_chunk_size = retrieval_chunk_size
        self.generation_kwargs = gen_kwargs
        self.mem_generation_kwargs = mem_gen_kwargs

        # hard coded to use gpt-3.5 tokenizer for convenience
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", self.retrieval_chunk_size)
        self.pre_cached = {}
        if pre_cached:
            self.pre_cached = pre_cached
        self.reset()

    def reset(self):
        if self.generator.model_name_or_path.find("memorag") != -1:
            self.generator.model.memory.reset()
        elif self.memory.model_name_or_path.find("memorag") != -1:
            self.memory.model.memory.reset()
            
        
    def __call__(self, context:str, question:str, prompt, cache_id="",task="", conv=True):
        """
        Recall relevant information from the memory module; 
        Based on both the question and the recalled memory, search for evidences in the context;
        Answer the question according to the recalled memory and evidences 
        """
        retrieval_query = []
        potential_answer = None
        if question:    
            if conv:
                results = self.memory.generate_conv(
                        question,
                        context,
                        _prompt_memoRAG,
                        [_instruct_sur, _instruct_span],
                        **self.mem_generation_kwargs
                        )
                for i,res in enumerate(results):
                    recall = res.split("\n")
                    retrieval_query.extend(recall)
                    if i == 2:
                        potential_answer = recall[0]
                        print("potential answer: ", potential_answer)
                    else:
                        print(f"{i}: \n", recall)
            else:
                raise NotImplementedError
            retrieval_query.append(question)
        else:
            raise NotImplementedError

        retrieval_query = [sent for sent in retrieval_query if len(sent.split()) > 3]
        retrieval_query = list(set(retrieval_query))
        
        if cache_id and cache_id in self.pre_cached:
            print(f"use pre-cached: {cache_id}")
            index = self.pre_cached[cache_id]["index"]
            retrieval_corpus = self.pre_cached[cache_id]["corpus"]
            self.retriever._index = index
            if len(retrieval_corpus) != index.index.ntotal:
                print(f"warning: {cache_id} has {len(retrieval_corpus)} chunks, but index has {index.index.ntotal} index")
                self.retriever.remove_all()
                retrieval_corpus = self.text_splitter.chunks(context)
                self.retriever.add(retrieval_corpus)
        else:
            retrieval_corpus = self.text_splitter.chunks(context)
            self.retriever.add(retrieval_corpus)

        # search for relevant context
        topk_scores, topk_indices = self.retriever.search(
            queries=retrieval_query)
        # reorder indices to keep consecutive chunks
        topk_indices = list(chain(*[topk_index.tolist() for topk_index in topk_indices]))
        topk_indices = list(set(topk_indices))

        topk_indices = sorted([x for x in topk_indices if x > -1])
        # slice out relevant context from the corpus
        retrieval_results = [retrieval_corpus[i].strip() for i in topk_indices]
        if potential_answer:
            retrieval_results.append(f"The answer might be:\n {potential_answer}.")
        knowledge = "\n\n".join(retrieval_results)

        if question:
            answer_prompt = prompt.format(input=question, context=knowledge)
        else:
            answer_prompt = prompt.format(context=knowledge)

        answer = self.generator.generate(answer_prompt, **self.generation_kwargs)
        self.reset()
        if cache_id not in self.pre_cached:
            self.retriever.remove_all()
        return answer

class HydePipeline:
    def __init__(self, 
        retriever: Union[DenseRetriever], 
        generator: Union[HuggingFaceModel], 
        gen_kwargs: Dict={},
        retrieval_chunk_size:int=512,
        pre_cached=None):
        self.pre_cached = {}
        if pre_cached:
            self.pre_cached = pre_cached

        self.retriever = retriever
        self.generator = generator

        self.retrieval_chunk_size = retrieval_chunk_size
        self.generation_kwargs = gen_kwargs

        # hard coded to use gpt-3.5 tokenizer for convenience
        self.text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", self.retrieval_chunk_size)

    def __call__(self, context, question, prompt, cache_id="", task="", conv=False):
        if cache_id and cache_id in self.pre_cached:
            print(f"use pre-cached: {cache_id}")
            index = self.pre_cached[cache_id]["index"]
            retrieval_corpus = self.pre_cached[cache_id]["corpus"]
            self.retriever._index = index
        else:
            retrieval_corpus = self.text_splitter.chunks(context)
            self.retriever.add(retrieval_corpus)

        hyde_prompt = query_rewrite.format(input=question)
        hyde_answer = self.generator.generate(hyde_prompt, **self.generation_kwargs)
        print(hyde_answer)
        topk_scores, topk_indices = self.retriever.search(
            queries=[question, hyde_answer])
        topk_indices = sorted([x for x in topk_indices[0].tolist() if x > -1])

        retrieval_results = [retrieval_corpus[i].strip() for i in topk_indices]
        knowledge = "\n\n".join(retrieval_results)

        if question:
            answer_prompt = prompt.format(input=question, context=knowledge)
        else:
            answer_prompt = prompt.format(context=knowledge)
        if not cache_id:
            self.retriever.remove_all()
        answer = self.generator.generate(answer_prompt, **self.generation_kwargs)
        return answer
