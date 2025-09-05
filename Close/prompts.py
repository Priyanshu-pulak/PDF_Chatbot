from langchain.prompts import PromptTemplate

NATURAL_QA_TEMPLATE = """You are an AI assistant helping users understand documents. Use the following context to answer the question in a natural, conversational manner.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer isn't in the context, say "I don't have enough information to answer that question."
- Write your response as complete, natural sentences - not as bullet points or lists
- Use conversational phrases like "According to the documents...", "The information shows that...", "Based on what I found..."
- Synthesize information from multiple sources into flowing paragraphs
- Avoid technical formatting or semicolon-separated lists
- Make your response sound like a knowledgeable person explaining the topic naturally

Answer:"""

TECHNICAL_QA_TEMPLATE = """You are a technical AI assistant helping users understand documents. Use the following context to answer the question with technical precision.

Context: {context}

Question: {question}

Instructions:
- Answer based only on the provided context
- If the answer isn't in the context, say "I don't have enough information to answer that question."
- Be specific and cite relevant details from the documents
- If multiple documents contain relevant information, synthesize the information clearly
- Use technical formatting when appropriate for clarity

Answer:"""

def get_natural_qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=NATURAL_QA_TEMPLATE,
        input_variables=["context", "question"]
    )

def get_technical_qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        template=TECHNICAL_QA_TEMPLATE,
        input_variables=["context", "question"]
    )

def get_default_qa_prompt() -> PromptTemplate:
    return get_natural_qa_prompt()
