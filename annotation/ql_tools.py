import streamlit as st
import tiktoken
import json
from collections import Counter, defaultdict
import os
from my_own_tools import *
from .utils import *
from .prompts import *
import requests
import random
@st.cache_data
def load_tokenizer():
    return tiktoken.encoding_for_model("gpt-3.5-turbo")

@st.cache_data
def load_context(path: str):
    with open(path, "r") as f:
        return json.load(f)

@st.cache_resource
def get_gpt4o(api_dict):
    return Agent("openai/gpt-4o", "openrouter", api_dict)

@st.cache_resource
def get_deepseek(api_dict):
    return Agent("deepseek-chat", "deepseek", api_dict)

@st.cache_resource
def get_deepseek_backup(api_dict):
    return Agent("deepseek-chat", "deepseek-backup", api_dict)

@st.cache_resource
def get_gpt35(api_dict):
    return Agent("gpt-35-turbo-16k", "azure", api_dict)

def get_agent(api_dict):
    with st.sidebar:
        agent_name = st.selectbox("Select Agent:", ["deepseek", "deepseek-backup", "gpt-4o"], key="agent_selector")
    
    if agent_name == "gpt-4o":
        return get_gpt4o(api_dict)
    elif agent_name == "deepseek-backup":
        return get_deepseek_backup(api_dict)
    else:
        return get_deepseek(api_dict)

def translate_text(agent, text: str, target_language: str="") -> str:
    prompt = f"Translate the following text to {target_language}: {text}, only output the translated text."
    return stream_output(agent, prompt, max_completion_tokens=512, format_json=False)

def get_annotated_counts(data_dir: str) -> dict:
    counts = {
        'total': 0,
        'by_level': Counter(),
        'by_domain': Counter(),
        'f1_score': defaultdict(list),
    }
    
    if not os.path.exists(data_dir):
        return counts
        
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.jsonl'):
                filepath = os.path.join(root, file)
                level = root.split("/")[-1]
                domain = file.split(".")[0]

                data = load_jsonl(filepath)
                for item in data:
                    counts['f1_score'][item["level"]].append(item["f1_score"])
                    # counts['gpt_score'][item["level"]].append(item["gpt_score"])
                counts['total'] += len(data)
                counts['by_level'][f"{level}"] += len(data)
                counts['by_domain'][domain] += len(data)
    return counts


def stream_output(agent, prompt: str, max_completion_tokens: int=512, format_json: bool=True):
    placeholder = st.empty()
    full_response = ""
    for token in agent.stream_completion(prompt, max_completion_tokens=max_completion_tokens):
        full_response += token
        if format_json:
            placeholder.markdown(f"```json\n{full_response}\n```")
        else:
            placeholder.markdown(full_response)
    return full_response

def retrieve_chunks(query: str, context: str, context_id: str, hits: int=5):
    response = requests.post(
                    "http://127.0.0.1:39112/search",
                    json={
                        "query": query,
                        "hits": hits,
                        "context": context,
                        "context_id": context_id
                }
            )
    if response.status_code == 200:
        results = response.json()
    else:
        st.error("Failed to retrieve similar chunks")
    return results

def get_rag_results(query: str, context: str, context_id: str, hits: int=5, generator: Agent=None, qa_prompt: str=None, chunks: list=None, answer: str=None):
    response = requests.post(
                    "http://127.0.0.1:39112/search",
                    json={
                        "query": query,
                        "hits": hits,
                        "context": context,
                        "context_id": context_id
                }
            )
    if response.status_code == 200:
        results = response.json()
    else:
        st.error("Failed to retrieve similar chunks")

    recall = len(set(results["chunks"]) & set(chunks)) / len(chunks)
    precision = len(set(results["chunks"]) & set(chunks)) / len(results["chunks"])
    st.write(f":green[**Recall:** {recall:.2f}]")
    st.write(f":green[**Precision:** {precision:.2f}]")
    retrieved_context = "\n\n".join(results["chunks"])

    rag_res = generator.chat_completion(qa_prompt.format(context=retrieved_context, input=query), max_completion_tokens=128)
    st.write("**RAG Result:**")
    st.write(rag_res)
    f1_score = qa_f1_score(rag_res, answer)
    st.write(f"**F1 Score:** {f1_score}")
                
    return rag_res, f1_score

def output_multi_choice(response: str):
    try:
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        qa_pair = json.loads(response)
        return qa_pair
    except json.JSONDecodeError:
        st.error("Failed to parse response as JSON")
        st.write("Raw response:")
        st.write(response)

def compute_multichoice_score(predicted_choices: list, reference_choices: list) -> float:
    predicted_set = set(predicted_choices)
    reference_set = set(reference_choices)
    
    if len(predicted_set - reference_set) > 0:
        return 0.0
    else:
        return len(predicted_set & reference_set) / len(reference_set)

def output_qa_pair(response: str, agent: Agent):
    try:
        # Clean response if wrapped in ```json ``` format
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]

        response = response.strip()
        qa_pair = json.loads(response)
        st.write("**Generated Question:**")
        st.write(qa_pair["question"])
        
        
        st.write("**Answer:**")
        st.write(qa_pair["answer"])
        st.write("**Generated QA in Chinese:**")
    except json.JSONDecodeError:
        print(response)
        st.error("Failed to parse response as JSON")
        st.write("Raw response:")
        st.write(response)
    
    return qa_pair

def get_gpt_score(agent, question: str, answer: str, rag_res: str):
    eval_prompt = ANSWER_EVALUATION_PROMPT.format(reference=answer, predicted=rag_res, question=question)
    gpt_score = agent.chat_completion(eval_prompt, max_completion_tokens=512)
    st.write(f"**GPT Score:** {gpt_score} / 9")
    return gpt_score

def sample_chunks(chunks: list, n_chunks: int):
    if len(chunks) < n_chunks:
        return chunks
    else:
        max_start = len(chunks) - n_chunks
        start_idx = random.randint(0, max_start)
        return chunks[start_idx:start_idx + n_chunks]
