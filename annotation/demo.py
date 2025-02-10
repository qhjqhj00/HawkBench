from my_own_tools import *

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from pathlib import Path
import tiktoken
import json
from collections import Counter, defaultdict
import random
from .prompts import *
from semantic_text_splitter import TextSplitter
import requests
import numpy as np
from .utils import *
from .ql_tools import *
from .pipeline import *
from datetime import datetime
api_dict = load_json("data/api_keys.json")



agent = get_agent(api_dict)
contexts = load_context("data/all_raw_context.json")
text_splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", 512)
tokenizer = load_tokenizer()
generator = get_gpt35(api_dict)

def main():
    st.title("Data Annotation Tool")
    # Add level selector in sidebar
    with st.sidebar:
        level = st.selectbox("Select Annotation Query Level:", [1, 2, 3, 4], key="level_selector")

    with st.sidebar:
        st.subheader("Annotation Stats")
        
        stats_dir = "data/omnibench_0106"
        counts = get_annotated_counts(stats_dir)
        st.write(f"**Total Annotations:** {counts['total']}")
        st.write("**Counts by Level:**")
        for _level, count in counts['by_level'].items():
            st.write(f"- {_level}: {count}")
        st.write("**Counts by Domain:**")
        for domain, count in counts['by_domain'].items():
            st.write(f"- {domain}: {count}")
        st.write("**F1 Score by Level:**")
        for _level, scores in counts['f1_score'].items():
            st.write(f"- {_level}: {np.mean(scores):.2f}")

    with st.sidebar:
        domain = st.selectbox("Select Domain:", ["all", "paper", "tech", "science", "humanities", "arts", "novel", "law", "finance"], key="domain_selector")
        if st.button("Sample New Context"):
            # Randomly sample a context
            if domain == "all":
                context_id = random.choice(list(contexts.keys()))
            else:
                context_id = random.choice([k for k, v in contexts.items() if v['domain'] == domain])
            context_data = contexts[context_id]
            context_text = context_data['context']
            
            # Split into chunks
            full_chunks = text_splitter.chunks(context_text)
            
            # Store in session state
            st.session_state.current_context = context_text
            st.session_state.current_chunks = full_chunks
            st.session_state.current_context_id = get_md5(context_text[:1024]+context_text[-1024:])
            st.session_state.current_context_data = context_data
            st.session_state.raw_context_id = context_id
            # Show success message
            st.success("New context sampled successfully!")
    # Initialize session state for storing context
    if 'current_context' not in st.session_state:
        st.session_state.current_context = None
        st.session_state.current_chunks = None
        st.session_state.current_context_id = None
        st.session_state.current_context_data = None
        st.session_state.raw_context_id = None
        st.session_state.result = None
    # Display current context and allow question generation if context exists
    if st.session_state.current_context is not None:
        st.subheader("Current Context")
        st.write(f"**Context ID:** {st.session_state.current_context_id}")
        st.write(f"**Domain:** {st.session_state.current_context_data['domain']}")
        st.write("**Meta Information:**")
        for key, value in st.session_state.current_context_data['meta'].items():
            st.write(f"- {key}: {value}")
        length = len(tokenizer.encode(st.session_state.current_context))
        st.write(f"**Context Length:** {length}")
        with st.expander("Show Context"):
            st.write(st.session_state.current_context)
        

        if st.button("Generate New Question"):
            if level == 1:
                qa_pair, rag_res, f1_score, gpt_score = level_1_pipeline(agent, generator, st.session_state.current_context, st.session_state.current_context_id, st.session_state.current_chunks)
            elif level == 2:
                qa_pair, rag_res, f1_score, gpt_score = level_2_pipeline(agent, generator, st.session_state.current_context, st.session_state.current_context_id, st.session_state.current_chunks)
            elif level == 3:
                qa_pair, rag_res, f1_score, multi_choice_score, gpt_score = level_3_pipeline(agent, generator, st.session_state.current_context, st.session_state.current_context_id, st.session_state.current_chunks)
            elif level == 4:
                qa_pair, rag_res, f1_score, multi_choice_score, gpt_score = level_4_pipeline(agent, generator, st.session_state.current_context, st.session_state.current_context_id, st.session_state.current_chunks)

            if level in [1, 2]:
                st.session_state.result = {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "sample_id": get_md5(qa_pair["question"]+qa_pair["answer"]),
                    "rag_answer": rag_res,
                    "f1_score": f1_score,
                    "gpt_score": gpt_score,
                    "level": level,
                    "context_id": st.session_state.raw_context_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            elif level in [3, 4]:
                st.session_state.result = {
                    "question": qa_pair["question"],
                    "answer": qa_pair["answer"],
                    "selections": qa_pair["selections"],
                    "selection_answers": qa_pair["selection_answers"],
                    "sample_id": get_md5(qa_pair["question"]+qa_pair["answer"]),
                    "rag_answer": rag_res,
                    "f1_score": f1_score,
                    "multi_choice_score": multi_choice_score,
                    "gpt_score": gpt_score,
                    "level": level,
                    "context_id": st.session_state.raw_context_id,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

        if st.session_state.result is not None:
            modified_answer = st.text_input("Modify answer if needed:", value=st.session_state.result["answer"])
            if modified_answer != st.session_state.result["answer"]:
                st.session_state.result["answer"] = modified_answer
            # Recalculate F1 score with modified answer
                f1_score = qa_f1_score(st.session_state.result["rag_answer"], modified_answer)
                st.write(f"**Updated F1 Score:** {f1_score}")
                st.session_state.result["f1_score"] = f1_score


        if st.button("Save", type="primary"):
            domain = st.session_state.current_context_data['domain']
            result = st.session_state.result
            if not os.path.exists(f"data/omnibench_0106/level_{level}"):
                os.makedirs(f"data/omnibench_0106/level_{level}")
            with open(f"data/omnibench_0106/level_{level}/{domain}.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            st.success("Results saved successfully!")       


if __name__ == "__main__":
    main()

