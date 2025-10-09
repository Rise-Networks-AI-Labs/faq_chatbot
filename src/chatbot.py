from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import os
from pydantic import BaseModel
from langchain.document_loaders.csv_loader import CSVLoader
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from typing import List
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain.schema import BaseRetriever, Document
from typing import List
from pydantic import Field

# Custom hybrid retriever
class HybridRetriever(BaseRetriever):
    kokokah_retriever: BaseRetriever = Field(...)
    tavily_retriever: BaseRetriever = Field(...)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        kokokah_docs = self.kokokah_retriever.get_relevant_documents(query)
        if kokokah_docs and len(kokokah_docs) > 0:
            return kokokah_docs
        return self.tavily_retriever.get_relevant_documents(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        kokokah_docs = await self.kokokah_retriever.aget_relevant_documents(query)
        if kokokah_docs and len(kokokah_docs) > 0:
            return kokokah_docs
        return await self.tavily_retriever.aget_relevant_documents(query)

# Get the base working directory
base_dir = os.path.dirname(os.path.abspath(__file__))
absolute_path = os.path.join(base_dir, "data", "kokokah_lms_faqs.csv")

print(f"📂 Current working directory: {os.getcwd()}")
print(f"📁 Expected CSV path: {absolute_path}")
print(f"📄 Exists? {os.path.exists(absolute_path)}")


# Load the environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Instantiating the fastapi app
app = FastAPI(title="FAQ Chatbot", version="0.1")


# welcome route
@app.get("/")
def load_app(name:str) -> dict:
    return {"Message": f"Welcome to FAQ Chatbot. Visit /docs to explore the API."}

# ask the chatbot questions
@app.post("/ask")
async def ask_chatbot(query: str):
    try:
        try:
            loader = CSVLoader(file_path=absolute_path, encoding="cp1252")
            documents = loader.load()
        except Exception as e:
            import traceback
            print("❌ Error loading CSV:", e)
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"CSV Load Error: {str(e)}")

        # Embeddings
        embedding_model = OpenAIEmbeddings(api_key=api_key)

        # FAISS store
        vectorstore = FAISS.from_documents(documents, embedding_model)
        kokokah_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Tavily retriever
        tavily_retriever = TavilySearchAPIRetriever(k=3)

        # Hybrid retriever
        hybrid_retriever = HybridRetriever(kokokah_retriever=kokokah_retriever, tavily_retriever=tavily_retriever)

        # LLM
        llm_model = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_retries=2,
        )

        # Prompt
        prompt_template = """
        You are an assistant for Kokokah. 
        Prefer using Kokokah's internal data first. 
        If no Kokokah info exists, then use external sources. 
        If neither has the answer, say:
        "I am sorry, I do not know the answer. Please reach out to customer support."

        {context}
        Question: {question}
        Answer:
        """
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )

        # RetrievalQA with hybrid retriever
        retrieval_qa = RetrievalQA.from_chain_type(
            llm=llm_model,
            retriever=hybrid_retriever,
            chain_type="stuff",
            chain_type_kwargs={
                "prompt": prompt,
                "memory": memory,
            },
            return_source_documents=True,
        )

        # Query
        response = retrieval_qa.invoke({"query": query})

        return response["result"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))