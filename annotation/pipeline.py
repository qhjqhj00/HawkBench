import streamlit as st
from .ql_tools import *
from .prompts import *

def level_1_pipeline(agent, generator, context, context_id, chunks):
    sampled_chunks = sample_chunks(chunks, 1)
    qg_prompt = QUESTION_GENERATION_PROMPT_L1  
    qa_prompt = QUESTION_ANSWERING_PROMPT_L1
    prompt = qg_prompt.format(text="\n\n".join(sampled_chunks))
    response = stream_output(agent, prompt, max_completion_tokens=512)
    qa_pair = output_qa_pair(response, agent)
    rag_res, f1_score = get_rag_results(qa_pair["question"], context, context_id, hits=5, qa_prompt=qa_prompt, generator=generator, chunks=sampled_chunks, answer=qa_pair["answer"])
    gpt_score = get_gpt_score(agent, qa_pair["question"], qa_pair["answer"], rag_res)
    return qa_pair, rag_res, f1_score, gpt_score

def level_2_pipeline(agent, generator, context, context_id, chunks):
    sampled_chunks = sample_chunks(chunks, 1)
    single_hop_qg_prompt = QUESTION_GENERATION_PROMPT_L1
    multi_hop_qg_prompt = QUESTION_GENERATION_PROMPT_L2
    qa_prompt = QUESTION_ANSWERING_PROMPT_L2

    st.write("Begin single-hop question generation")
    prompt = single_hop_qg_prompt.format(text="\n\n".join(sampled_chunks))
    response = stream_output(agent, prompt, max_completion_tokens=512)
    qa_pair = output_qa_pair(response, agent)

    st.write("Begin multi-hop question generation")
    relevant_chunks = retrieve_chunks(qa_pair["question"], context, context_id, hits=10)["chunks"]

    context = ""
    for i, chunk in enumerate(relevant_chunks):
        context += f"Passage {i}: {chunk}\n\n"
    prompt = multi_hop_qg_prompt.format(query=qa_pair["question"], chunks=context)
    response = stream_output(agent, prompt, max_completion_tokens=512)
    qa_pair = output_qa_pair(response, agent)

    rag_res, f1_score = get_rag_results(qa_pair["question"], context, context_id, hits=5, qa_prompt=qa_prompt, generator=generator, chunks=relevant_chunks, answer=qa_pair["answer"])
    gpt_score = get_gpt_score(agent, qa_pair["question"], qa_pair["answer"], rag_res)
    return qa_pair, rag_res, f1_score, gpt_score

def level_3_pipeline(agent, generator, context, context_id, chunks):
    sampled_chunks = sample_chunks(chunks, 100)
    qg_prompt = QUESTION_GENERATION_PROMPT_L3
    qa_prompt = QUESTION_ANSWERING_PROMPT_L3

    prompt = qg_prompt.format(text="\n".join(sampled_chunks))
    response = stream_output(agent, prompt, max_completion_tokens=512)
    qa_pair = output_qa_pair(response, agent)

    answer_decom_prompt = ANSWER_DECOMPOSITION_PROMPT_L4
    prompt = answer_decom_prompt.format(question=qa_pair["question"], answer=qa_pair["answer"])
    response = stream_output(agent, prompt, max_completion_tokens=512)
    answer_decom = output_multi_choice(response)

    qa_pair["selections"] = answer_decom["selections"]
    qa_pair["selection_answers"] = answer_decom["answers"]

    rag_res, f1_score = get_rag_results(
        qa_pair["question"], context, context_id, hits=5, qa_prompt=qa_prompt, generator=generator, chunks=sampled_chunks, answer=qa_pair["answer"])

    str_selections = ""
    for k, v in qa_pair["selections"].items():
        str_selections += f"{k}. {v}\n"
    multi_choice_prompt = ANSWER_MULTICHOICE_PROMPT_L4
    prompt = multi_choice_prompt.format(
        question=qa_pair["question"], selections=str_selections, answer=rag_res)

    response = generator.chat_completion(prompt, max_completion_tokens=512)
    st.write(response)
    try:
        multi_choice_score = compute_multichoice_score(
            json.loads(response), qa_pair["selection_answers"])
        st.write(f"**Multi-Choice Score:** {multi_choice_score}")
    except:
        multi_choice_score = 0

    gpt_score = get_gpt_score(agent, qa_pair["question"], qa_pair["answer"], rag_res)

    return qa_pair, rag_res, f1_score, multi_choice_score, gpt_score

def level_4_pipeline(agent, generator, context, context_id, chunks):
    sampled_chunks = sample_chunks(chunks, 100)
    qg_prompt = QUESTION_GENERATION_PROMPT_L4
    qa_prompt = QUESTION_ANSWERING_PROMPT_L4

    prompt = qg_prompt.format(text="\n".join(sampled_chunks))
    response = stream_output(agent, prompt, max_completion_tokens=512)
    qa_pair = output_qa_pair(response, agent)

    answer_decom_prompt = ANSWER_DECOMPOSITION_PROMPT_L4
    prompt = answer_decom_prompt.format(question=qa_pair["question"], answer=qa_pair["answer"])
    response = stream_output(agent, prompt, max_completion_tokens=512)
    answer_decom = output_multi_choice(response)

    qa_pair["selections"] = answer_decom["selections"]
    qa_pair["selection_answers"] = answer_decom["answers"]

    rag_res, f1_score = get_rag_results(
        qa_pair["question"], context, context_id, hits=5, qa_prompt=qa_prompt, generator=generator, chunks=sampled_chunks, answer=qa_pair["answer"])

    str_selections = ""
    for k, v in qa_pair["selections"].items():
        str_selections += f"{k}. {v}\n"
    multi_choice_prompt = ANSWER_MULTICHOICE_PROMPT_L4
    prompt = multi_choice_prompt.format(
        selections=str_selections, answer=rag_res)

    response = generator.chat_completion(prompt, max_completion_tokens=512)
    st.write(response)
    try:    
        multi_choice_score = compute_multichoice_score(
            json.loads(response), qa_pair["selection_answers"])
        st.write(f"**Multi-Choice Score:** {multi_choice_score}")
    except:
        multi_choice_score = 0

    gpt_score = get_gpt_score(agent, qa_pair["question"], qa_pair["answer"], rag_res)

    return qa_pair, rag_res, f1_score, multi_choice_score, gpt_score