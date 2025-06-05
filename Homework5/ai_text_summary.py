from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv
import os

load_dotenv()
user_agent = os.getenv("USER_AGENT")
api_key = os.getenv("GEMINI_API_KEY")

os.environ["USER_AGENT"] = user_agent

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=api_key)
loader = WebBaseLoader("https://habr.com/ru/companies/cian/articles/892650/")
docs = loader.load()

prompt = ChatPromptTemplate.from_template("Напишите краткое изложение следующего текста: {context}")
chain = create_stuff_documents_chain(llm, prompt)

try:
    result = chain.invoke({"context": docs})
    print("=" * 50)
    for text in result.split('.'):
        print(text)
    print("=" * 50)
except Exception as e:
    print(f"Ошибка: {e}")