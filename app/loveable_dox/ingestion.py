import asyncio
import ssl
import os
import pickle
from typing import Any, Dict, List
from dotenv import load_dotenv
import certifi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone.vectorstores import PineconeVectorStore
from langchain_tavily import TavilyCrawl
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from app.loveable_dox.clean_docs import clean_docs_folder
from app.loveable_dox.index_docs import run


load_dotenv()


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUEST_CA_BUNDLE'] = certifi.where()

tavily_crawl = TavilyCrawl()


async def ingestion():
    print('Ingestion is starting...')
    print('Crawling https://docs.lovable.dev/tips-tricks/best-practice')

    # result = tavily_crawl({
    #     "max_depth": 2,
        # url: multiple loveable urls
    # })

    # all_docs = [Document(page_content=result["raw_content"], metadata={"source": result["url"]}) for result in res["results"]]
    # print(len(all_docs))

    # with open("tips-tricks-best-practice_raw.txt", "w", encoding="utf-8") as f:
    #     f.write(str(all_docs))
    # with open("tips-tricks-best-practice_crawl_result.pkl", "wb") as f:
    #     pickle.dump(all_docs, f)
    # print('Saved')

    base_path = Path(__file__).parent.parent.parent / "11_02_2026_pickle"
    
    # cleaned = clean_docs_folder(base_path, output_folder="./cleaned")

    # cleaned_path = Path(r"C:\data\discord_automate_sales\v4\cleaned")
    # all_docs = []
    # for filepath in sorted(cleaned_path.glob("*.pkl")):
    #     with open(filepath, "rb") as f:
    #         all_docs.extend(pickle.load(f))

    # print(f"Dokumentów do embeddingu: {len(all_docs)}")
    vectorstore = run(
        cleaned_folder=r"C:\data\discord_automate_sales\v4\cleaned",
        chroma_dir=r"C:\data\discord_automate_sales\v4\chroma_db",  # domyślnie ../chroma_db
        chunk_size=512,       # domyślnie 512
        chunk_overlap=64,     # domyślnie 64
        model_name="BAAI/bge-small-en-v1.5",  # domyślnie bge-small
        batch_size=256,       # domyślnie 256
        skip_sanity=False,    # domyślnie False — uruchomi 3 test queries
    )


if __name__ == "__main__":
    asyncio.run(ingestion())