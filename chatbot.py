import os
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from langchain import hub
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from openai import OpenAI

#INITIALIZE THE COMPONENTS
PINECONE_API_KEY = os.environ.get("PINECONEAPIKEY")
pc = Pinecone(api_key = PINECONE_API_KEY)
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1"
)

loader = WebBaseLoader(
    web_path = "https://www.adept-techno.com", 
    bs_kwargs = dict(                          
        parse_only = bs4.SoupStrainer(
            class_ = "page-content"
        )
    )
)
docs = loader.load()

#SPLIT THE DOCUMENT
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 200,
)
all_splits = text_splitter.split_documents(docs)

#GENERATE EMBEDDINGS
embeddings = pc.inference.embed(
    model = "multilingual-e5-large",
    inputs = [split.page_content for split in all_splits],
    parameters = {
        "input_type" : "passage",
        "truncate" : "END"
    }
)

#STORE EMBEDDINGS IN VECTOR DATABASE
index = pc.Index("chatbot")
vectors_to_upsert = []
number_of_embeddings = 0

for embedding in embeddings.data:
    id = f"Chunk {number_of_embeddings}"
    values = embedding["values"]
    metadata = {
        "source" : all_splits[number_of_embeddings].metadata["source"],
        "content" : all_splits[number_of_embeddings].page_content
    }
    vectors_to_upsert.append({
        "id" : id,
        "values" : values,
        "metadata" : metadata
    })
    number_of_embeddings += 1

try:
    index.upsert(
        vectors = vectors_to_upsert,
        namespace = "ns1"
    )
    print("Data upserted successfully!")
except Exception as e:
    print(f"It did not work. Here's why: {e}")

#PROMPT TEMPLATE FOR THE LLM (https://smith.langchain.com/hub/rlm/rag-prompt) 
prompt = hub.pull("rlm/rag-prompt")
refined_prompt = prompt.messages[0].prompt.template

#DEFINE STATE FOR APPLICATION 
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

#GENERATE EMBEDDINGS FOR USER'S QUESTION
def retrieve(state: State):
   embedded_question = pc.inference.embed(
    model = "multilingual-e5-large",
    inputs = state["question"],
    parameters = {
        "input_type" : "query",
        "truncate" : "END"
    }
   )
   embedded_question = embedded_question.data[0]["values"]

   #QUERY THE DATABASE. RESULT IS CONTEXT FOR LLM
   results = index.query(
    namespace = "ns1",
    vector = embedded_question,
    top_k = 3,
    include_metadata = True,
    include_values = False
   )
   context = []
   for result in results["matches"]:
    context.append(Document(
        page_content = result["metadata"]["content"],
        metadata = {
            "source" : result["metadata"]["source"]
        }
    ))

    return context
    
#GENERATE AN ANSWER FROM THE LLM
def generate(state: State):
    context = retrieve(state)
    state["context"] = context
    context_str = "\n".join(doc.page_content for doc in state["context"])
    user_message = f"Question: {state['question']}\nContext: {context_str}"

    response = client.chat.completions.create(
        model = "deepseek-chat",
        messages = [
            {"role": "system", "content": refined_prompt},
            {"role": "user", "content": user_message}
        ], 
        stream = False
    )

    state["answer"] = response.choices[0].message.content
    return state["answer"]

state = {
    "question": "where is adept located?",
    "context": [],
    "answer": ""
}

state = generate(state)

print("Answer:", state)
