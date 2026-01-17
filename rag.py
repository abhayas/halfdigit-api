import os
import langchain
import datetime


# 1. IMPORTS
# Loaders & Splitters
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Vector Store & AI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# Core Components (LCEL)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

# --- CONFIGURATION ---
# Make sure your API key is set in your environment variables
#os.environ["OPENAI_API_KEY"] = "sk-..." 
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print(f"LangChain Version: {langchain.__version__}")

# --- 1. LOAD DATA ---
print("Loading knowledge base...")

# List of files to load
loaders = [
    TextLoader("knowledge/profile.txt", encoding="utf-8"),
    TextLoader("knowledge/projects.txt", encoding="utf-8"),
    TextLoader("knowledge/resume.txt", encoding="utf-8"),
]

docs = []
for loader in loaders:
    try:
        docs.extend(loader.load())
        print(f"Loaded: {loader.file_path}")
    except Exception as e:
        print(f"WARNING: Could not load {loader.file_path}. Error: {e}")

if not docs:
    print("CRITICAL ERROR: No documents loaded. Exiting.")
    exit()

# --- 2. SPLIT DOCUMENTS ---
# Splitting long text into smaller chunks helps the AI find specific details
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} knowledge chunks.")

# --- 3. CREATE VECTOR STORE ---
print("Creating vector database...")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embedding)
retriever = vectorstore.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

today_date = datetime.date.today()
system_prompt = (
    "You are a helpful AI assistant for Abhaya Prasad Sahu's portfolio. "
    "refer Abhaya Prasad Sahu as Abhaya in response"
    "Your main job is to answer questions based on his resume, profile and projects. "
    "Experience calculations need to be done as per todays date {today_date}."
    "\n\n"
    "GUIDELINES:\n"
    "1. If the user says 'hi', 'hello', or asks 'who are you', answer politely and introduce yourself.\n"
    "2. For specific questions about Abhaya (experience, skills, projects), use the Context below.\n"
    "3. If the answer is NOT in the context, say: 'I don't have that information in my current knowledge base. Please contact Abhaya for more details'\n"
    "\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough(),"today_date": lambda _: datetime.date.today()}
    | prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    while True:
        try:
            query = input("\nYou: ")
            if query.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            if not query.strip():
                continue

            print("Bot: Thinking...")
            response = rag_chain.invoke(query)
            print(f"Bot: {response}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")