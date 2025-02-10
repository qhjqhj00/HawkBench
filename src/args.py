import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict

@dataclass
class ModelArgs:
    model_cache_dir: str = field(
        default='/share/shared_models',
        metadata={'help': 'Default path to save language models.'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side.'}
    )
    access_token: Optional[str] = field(
        default="",
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: str = field(
        default="flash_attention_2",
        metadata={'help': 'The implementation of attention.'}
    )
    index_path: str = field(
        default="",
        metadata={'help': 'Path to pre-cached index.'}
    )
    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    chat_template: str = field(
        default="llama-2",
        metadata={'help': 'Instruction template name in fastchat.'}
    )
    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )
    cache_implementation: str = field(
        default=None,
        metadata={'help': 'use cache?'}
    )

    cache_backend: str = field(
        default=None,
        metadata={'help': 'cache backend'}
    )

    cache_nbits: int = field(
        default=None,
        metadata={'help': 'quant size'}
    )

    load_in_4bit: bool = field(
        default=False,
        metadata={'help': 'quant size'}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )
    ret_model: str = field(
        default="BAAI/bge-m3",
        metadata={'help': 'Model name or path for retrieval.'}
    )
    ret_dense_pooling: str = field(
        default="cls",
        metadata={'help': 'Pooling method for dense retrieval model. {mean, cls}'}
    )
    ret_dense_q_max_len: int = field(
        default=256,
        metadata={'help': 'Query max length for dense retrieval model.'}
    )
    ret_dense_k_max_len: int = field(
        default=512,
        metadata={'help': 'Key max length for dense retrieval model.'}
    )
    ret_dense_metric: str = field(
        default="cos",
        metadata={'help': 'Similarity metric for dense retrieval model.'}
    )
    ret_hits: int = field(
        default=3,
        metadata={'help': 'Number of retrieval candidates.'}
    )
    ret_chunk_size: int = field(
        default=256,
        metadata={'help': 'Number of tokens within each chunk for retrieval.'}
    )
    gen_model: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={'help': 'Model name or path for generation.'}
    )
    mem_model: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.2",
        metadata={'help': 'Model name or path for generation.'}
    )
    use_minference: bool = field(
        default=False,
        metadata={'help': 'Use minference?'}
    )
    gen_max_new_tokens: Optional[int] = field(
        default=512,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    gen_do_sample: Optional[bool] = field(
        default=False,
        metadata={'help': 'Do sampling when decoding?'},
    )
    gen_temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    gen_top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    pipeline: str = field(
        default="",
        metadata={'help': ''}
    )
    note: str = field(
        default="",
        metadata={'help': 'experiment note'}
    )
    conv: bool = field(
        default=False,
        metadata={'help': 'Merge and unload LoRA?'},
    )
    api_dict: str = field(
        default="",
    )
    agent_provider : str = field(
        default="",
    )
    agent_model : str = field(
        default="",
    )
    use_api: bool = field(
        default=False,
    )
    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)
