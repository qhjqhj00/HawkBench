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
Answer: ...

Here is the text:
{context}

Question: {input}

Your response:"""

QUESTION_ANSWERING_PROMPT = """You are a question answering assistant. Your task is to answer questions based solely on the provided text.

Follow these rules:
1. Only use information provided in the given text, do not make up information
2. Provide direct, concise answers without explanations
3. For simple questions, use short phrases when possible
4. For complex questions, make the answer as concise as possible, avoid repeating words from the question

Example 1 (simple question):
Question: What was the total revenue generated from international sales in Q2 2023?
Answer: $847 million

Example 2 (complex question):
Question: How do the relationships and interactions between marine species and human activities reflect broader challenges in ocean conservation efforts?
Answer: Marine species like turtles and jellies face complex threats from human exploitation and environmental changes, highlighting the intricate challenges of balancing conservation with economic interests.

Here is the text:
{context}

Question: {input}

Your response:"""

ANSWER_EVALUATION_PROMPT = """You are an evaluator assessing the semantic equivalence between a predicted answer and a reference answer. Your task is to assign a score from 0 to 9 based on these strict criteria.

Scoring criteria:
0: The predicted answer contains different or contradictory information compared to the reference answer
1-3: The predicted answer discusses related but different aspects than the reference answer, making them semantically distinct
4-6: The predicted answer has similar high-level meaning but differs in specific details or implications
7-8: The predicted answer conveys nearly the same meaning with minor differences in phrasing or emphasis
9: The predicted answer is semantically identical to the reference answer, expressing exactly the same meaning

Example:
Reference Answer: The company's revenue increased by 25% in Q4 2023 due to strong holiday sales.
Predicted Answer: The company saw a 25% revenue growth in the fourth quarter of 2023 because of high sales during holidays.
Score: 9 (Expresses exactly the same meaning with only superficial rewording)

Example:
Reference Answer: The Industrial Revolution led to rapid urbanization, technological advancement, and significant social changes in European society during the 19th century.
Predicted Answer: The Industrial Revolution brought economic growth and factory systems to Europe.
Score: 3 (Discusses related but different aspects - economic/industrial changes vs societal/technological changes)

Reference Answer: {reference}
Predicted Answer: {predicted}

Provide only the numerical score (0-9):"""

ANSWER_MULTICHOICE_PROMPT = """You are a multiple-choice question answering assistant. Your task is to analyze what selections are correct and incorrect based on the predicted answer.

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

ANSWER_CONTAINS_PROMPT = """You are an evaluator assessing whether a sentence is semantically contained within a longer reference answer. Your task is to determine if the key meaning and information of the sentence is fully expressed somewhere within the reference answer, and that the sentence does not contain any additional information beyond what is in the reference.

Follow these rules:
1. The sentence must be fully semantically contained - all key information and implications must be present in the reference
2. The sentence must not contain any information or implications that go beyond what is stated in the reference
3. The information can be spread across multiple sentences in the reference
4. Different wording or phrasing is acceptable as long as the core meaning is preserved
5. Provide only 1 / 0 as your response

Example:
Reference Answer: The Industrial Revolution began in Britain in the late 18th century. It led to major technological advances in manufacturing and transportation. This period saw rapid urbanization as people moved to cities for factory work.

Sentence to check: People moved from rural areas to cities to work in factories during the Industrial Revolution.
Response: 1

Sentence to check: The Industrial Revolution caused urbanization and social upheaval.
Response: 0 (Contains information about social upheaval not mentioned in reference)

Reference Answer: {reference}
Sentence to check: {sentence}

Provide only 1 / 0:"""

_prompt_memoRAG = """You are provided with a long article. Read the article carefully. After reading, you will be asked to perform specific tasks based on the content of the article.

Now, the article begins:
- **Article Content:** {context}

The article ends here.

Next, follow the instructions provided to complete the tasks."""

_instruct_sur = """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to generate precise clue questions that can help locate the necessary information.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of precise surrogate questions that clarify the original question.
3. Only output the clues. If there are multiple clues, separate them with a newline."""

_instruct_span = """
You are given a question related to the article. To answer it effectively, you need to recall specific details from the article. Your task is to identify and extract one or more specific clue texts from the article that are relevant to the question.

### Question: {question}
### Instructions:
1. You have a general understanding of the article. Your task is to generate one or more specific clues that will help in searching for supporting evidence within the article.
2. The clues are in the form of text spans that will assist in answering the question.
3. Only output the clues. If there are multiple clues, separate them with a newline."""


_instruct_qa = """
You are given a question related to the article. Your task is to answer the question directly.

### Question: {question}
### Instructions:
Provide a direct answer to the question based on the article's content. Do not include any additional text beyond the answer."""

hyde = "Please write a passage to answer the question.\nQuestion: {input}\nPassage:"
query_rewrite= "Please rewrite the question to be more specific.\nQuestion: {input}\nRewritten Question:"
