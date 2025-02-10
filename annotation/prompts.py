QUESTION_GENERATION_PROMPT_L1 = """You are a question generation assistant. Your task is to generate a question based on the given text chunk. The question should be answerable using a specific text span within the text chunk.

Follow these rules:
1. Generate a question that can be answered directly from the text
2. The answer should be a short, specific text span from the original text
3. The question should be clear and unambiguous
4. Avoid questions that require reasoning across multiple parts of the text
5. Output the question and answer in JSON format with "question" and "answer" fields

Here is the text chunk:
{text}

Generate a question and answer pair in JSON format.

Example output:
{{
    "question": "What was the capital city mentioned in the text?", 
    "answer": "Paris"
}}

Your response:"""



QUESTION_GENERATION_PROMPT_L2 = """You are a question generation assistant. Your task is to generate a complex multi-hop question based on a given single-hop query and relevant text chunks. The question should require at least three logical steps of reasoning across different pieces of information to arrive at the answer.

Follow these rules:
1. Generate a question that requires connecting at least 3 different facts from the text in a logical sequence
2. Each reasoning step should build on previous steps to reach the final answer
3. The answer should be short, concise and specific, try to answer the question in a single phrase or short sentence, avoiding repetition of question words
4. The question should be concise and clear and unambiguous despite its complexity, avoid combine multiple questions in one
5. All information needed to answer must be present in the provided chunks
6. Output the question and answer in JSON format

Here is the initial query and relevant text chunks:
Query: {query}
Relevant chunks:
{chunks}

Generate a complex multi-hop question and answer in JSON format.

Example output:
{{
    "question": "If the CEO who founded Company X in 1995 invested their initial salary into Company Y's stock in 2000, what would be the value of that investment today given the reported growth rate?",
    "answer": "$2.4 million"
}}

Your response:"""

QUESTION_GENERATION_PROMPT_L3 = """You are a question generation assistant. Your task is to generate a context-dependent analytical question that requires understanding and synthesizing distributed evidence across the full context. The question should focus on aspects that:

- Require global understanding of the context to be properly interpreted
- Need evidence scattered across different parts of the text
- Have explicit supporting rationale present in the context

Follow these rules:
1. Generate a question that cannot be answered without understanding the full context
2. The question should require finding and connecting multiple pieces of evidence
3. All evidence and rationale needed must exist explicitly in the context
4. The answer should be concise, short, and avoid repeating words from the question
5. Output both the context-dependent question and its answer in JSON format

Here is the text:
{text}

Generate a context-dependent analytical question and answer in JSON format.

Example output:
{{
    "question": "What measures did the company take during the pandemic to prevent its cash flow from breaking down?",
    "answer": "The company implemented three key measures: 1) Reduced operational costs by 30% through temporary facility closures, supported by evidence of shutdown dates and cost savings, 2) Secured additional credit lines of $50M as shown in the financial statements, 3) Accelerated digital transformation initiatives that maintained 85% of regular revenue streams, as demonstrated by the Q2-Q4 sales data"
}}

Your response:"""


QUESTION_GENERATION_PROMPT_L4 = """You are a question generation assistant. Your task is to generate a high-level analytical question that requires synthesizing broad information and understanding global implications from the given text. The question should focus on high-level aspects like:

- Key themes and motifs
- Underlying patterns and trends
- Societal or cultural impacts
- Historical significance
- Philosophical implications
- Systemic changes and transformations
- Broader context and connections
- Future trajectories and possibilities

You can also mine other high-level aspects from the text.

Follow these rules:
1. Generate a question that requires aggregating and analyzing information across the entire text
2. The question should be concise and high-level
3. The answer should be concise, and avoid repeating words from the question
4. The question should focus on broader impacts, evolution of trends, or systemic changes etc.
5. The answer should require understanding complex relationships and long-term implications
6. Output the question, answer in JSON format

Here is the text:
{text}

Generate a high-level analytical question, answer in JSON format.

Example output:
{{
    "question": "How has the company's strategic focus evolved over the past 5 years and what are its implications for the industry?",
    "answer": "..."
}}

Your response:"""

ANSWER_DECOMPOSITION_PROMPT_L4 = """You are a question decomposition assistant. Your task is to transform a high-level analytical answer into multiple-choice selections.

Follow these rules:
1. Break down the answer into 1-5 short statements, these statements can be induced from the answer with some reasoning
2. Make out 3-5 incorrect statements, these statements should be related to the question but contain subtle flaws or biases, cannot be supported by the answer. The incorrect statesments should confuse the reader and make it hard to choose the correct answer
3. Format the output as a JSON with letter-based selections and correct answers

Example:
Question: How has the rise of remote work impacted organizational culture?
Answer: Companies have shifted toward digital collaboration and flexible schedules, fundamentally changing workplace dynamics and employee expectations

Output:
{{
    "selections": {{
        "a": "...",
        "b": "...",
        "c": "...",
        "d": "...",
        "e": "..."
    }},
    "answers": [...]
}}


Here is the question and answer:
Question: {question}
Answer: {answer}

Transform this into multiple-choice format:"""


QUESTION_ANSWERING_PROMPT_L1 = """You are a question answering assistant. Your task is to answer a question based on the given text. The answer should be found directly within the text.

Follow these rules:
1. The answer should be a short, specific text span from the original text
2. If the answer cannot be found in the text, respond with "Cannot be answered from the given text"
3. Only use the knowledge from the text to answer the question, do not make up information
4. Do not provide explanations or reasoning
5. Only output the answer

Example:
Question: What is the capital of France?
Answer: Paris

Here is the text:
{context}

Question: {input}

Your response:"""

QUESTION_ANSWERING_PROMPT_L2 = """You are a question answering assistant. Your task is to answer a multi-hop question that requires connecting multiple pieces of information from the given text.

Follow these rules:
1. The answer should be derived by connecting specific text spans
2. If the answer cannot be supported by the text, respond with "Cannot be answered from the given text"
3. Only use the knowledge from the text to answer the question, do not make up information
4. Do not provide explanations or reasoning
5. Only output the answer

Example:
Question: If the CEO who founded Company X in 1995 invested their initial salary into Company Y's stock in 2000, what would be the value of that investment today given the reported growth rate?
Answer: $2.4 million

Here is the text:
{context}

Question: {input}

Your response:"""

QUESTION_ANSWERING_PROMPT_L3 = """You are a question answering assistant. Your task is to provide a reasoned answer to an analytical question based on the given text.

Follow these rules:
1. Analyze and interpret the information provided in the text
2. The answer should be a concise and clear, avoid repeating words from the question
3. Only use the knowledge from the text to answer the question, do not make up information
4. Do not provide explanations or reasoning
5. Only output the answer

Example:
Question: What measures did the company take during the pandemic to prevent its cash flow from breaking down?
Answer: The company implemented three key measures: 1) Reduced operational costs by 30% through temporary facility closures, supported by evidence of shutdown dates and cost savings, 2) Secured additional credit lines of $50M as shown in the financial statements, 3) Accelerated digital transformation initiatives that maintained 85% of regular revenue streams, as demonstrated by the Q2-Q4 sales data

Here is the text:
{context}

Question: {input}

Your response:"""

QUESTION_ANSWERING_PROMPT_L4 = """You are a question answering assistant. Your task is to provide a comprehensive answer to questions requiring high-level information synthesis and global understanding of themes, significance, or impact.

Follow these rules:
1. Analyze broad themes, patterns and implications across the entire text
2. The answer should synthesize information from multiple parts of the text to draw meaningful conclusions
3. Provide a clear answer that addresses the high-level question, make the answer as concise as possible, avoid repeating words from the question
4. Only use the knowledge from the text to answer the question, do not make up information
5. Do not provide explanations or reasoning
6. Only output the answer

Example:
Question: How do the relationships and interactions between marine species and human activities reflect broader challenges in ocean conservation efforts?
Answer: Marine species like turtles and jellies face complex threats from human exploitation and environmental changes, highlighting the intricate challenges of balancing conservation with economic interests.

Here is the text:
{context}

Question: {input}

Your response:"""

ANSWER_EVALUATION_PROMPT = """You are an evaluator assessing both the faithfulness between a predicted answer and a reference answer, and whether the predicted answer properly addresses the question. Your task is to assign a score from 0 to 9 based on these criteria.

Scoring criteria:
0-2: The predicted answer is completely incorrect, contradicts the reference answer, or fails to address the question
3-4: The predicted answer either captures only a small portion of the reference answer's meaning, contains significant inaccuracies, or only partially addresses the question
5-6: The predicted answer captures the main idea but misses important details, contains minor inaccuracies, or doesn't fully address all aspects of the question
7-8: The predicted answer accurately reflects most aspects of the reference answer and adequately addresses the question with minimal omissions
9: The predicted answer perfectly captures the complete meaning of the reference answer and fully addresses all aspects of the question

Example:
Question: What was the company's revenue growth in Q4 2023?
Reference Answer: The company's revenue increased by 25% in Q4 2023 due to strong holiday sales.
Predicted Answer: The company saw a 25% revenue growth in the fourth quarter of 2023 because of high sales during holidays.
Score: 9 (Perfect semantic match with all key information preserved and directly answers the question)

Question: {question}
Reference Answer: {reference}
Predicted Answer: {predicted}

Provide only the numerical score (0-9):"""

ANSWER_MULTICHOICE_PROMPT_L4 = """You are a multiple-choice question answering assistant. Your task is to analyze what selections are correct and incorrect based on the predicted answer.

Follow these rules:
1. Carefully analyze each option provided in the selections
2. Consider the predicted answer and any relevant information provided
3. Provide your answer as a list of correct letter choices

Example:
Predicted Answer: Urbanization leads to habitat fragmentation, higher pollution levels, and changes in local temperature patterns
Selections:
a. Increased habitat fragmentation
b. Improved biodiversity in all cases
c. Higher levels of pollution
d. Complete elimination of wildlife
e. Changes in local temperature patterns

Answer: ["a", "c", "e"]

Here is your task:
Predicted Answer: {answer}
Selections: 
{selections}

Provide your answer, only output the letter choices:"""



