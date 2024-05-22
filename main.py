from langchain_community.llms import Ollama
from langchain_community.document_loaders import TextLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)

# Change the model parameter for prefered llm
llm = Ollama(model="openhermes")
embbed = OllamaEmbeddings(model="nomic-embed-text")

# Load pdf document
loader = PyPDFDirectoryLoader("your directory")
doc = loader.load_and_split()

# Load text file
#loader = TextLoader("./the_great_gatsby.txt")
#doc = loader.load()
    
# Spliting the text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(doc)
    
# Index the text chunks
vectorstore = Chroma.from_documents(documents=splits, embedding=embbed)

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

# Use default prompt from langchain for rag apps
#prompt = hub.pull("rlm/rag-prompt")

# Format the response
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

"""
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
"""

# Contextualize question
contextualize_q_system_prompt = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_system_prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

config = {"configurable": {"session_id": "abc123"}}

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", " "],
    keep_separator=False,
)
compressor = EmbeddingsFilter(embeddings=embbed, k=10)

def split_and_filter(input):
    docs = input["docs"]
    question = input["question"]
    split_docs = splitter.split_documents(docs)
    stateful_docs = compressor.compress_documents(split_docs, question)
    return [stateful_doc for stateful_doc in stateful_docs]

retrieve = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever) | split_and_filter
)

def generate_response(query:str):
    answer = conversational_rag_chain.invoke({"input": query}, config=config)
    print(answer["answer"])
    print("")
    docs = retrieve.invoke(query)
    print("CONTEXT\n")
    for doc in docs:
        print(doc.page_content)

    """
    for chunk in conversational_rag_chain.stream({"input": query}, config={"configurable": {"session_id": "abc123"}},):
        if answer_chunk := chunk.get("answer"):
            print(f"{answer_chunk}", end="", flush=True)
        #print("")
    """

    
# cleanup the vectorstore
#vectorstore.delete_collection()

if __name__ == "__main__":
    print("How can i help you?")
    while True:
        query = input()
        if query == 1:
            break
        else:
            generate_response(query)
