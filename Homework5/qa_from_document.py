import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

loader = PyPDFLoader('romeo-and-juliet.pdf')
pages = loader.load()

vector_store = InMemoryVectorStore.from_documents(
    pages,
    GoogleGenerativeAIEmbeddings(model="models/embedding-001")
)

query = "Romeo gets poisoned"
docs = vector_store.similarity_search(query, k=2)

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
qa_chain = load_qa_chain(llm, chain_type="stuff")
response = qa_chain.run(input_documents=docs, question=query)

print('=' * 30)
print(f"Вопрос: {query}")
print(f"Ответ: {response}")
print('=' * 30)

for doc in docs:
    print('=' * 30)
    print(f'Page {doc.metadata["page"]}: {doc.page_content}\n')
    print('+' * 30)