from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import State
from app.graph.nodes.process_rag.retriever_openai_embed import Retriever
from config import get_openai
from app.graph.nodes.process_rag.prompt import INSIGHT_PROMPT

r = Retriever(
    score_threshold=0.3,
    final_k=5,)
llm = get_openai()


async def process_rag(state: State) -> State:
    query = state["lead_judge"].devdocs_query
    if not query:
        return {
            "rag_insight": None
        }
    context = r.search(query)
    prompt = ChatPromptTemplate.from_messages(
        [
            ('system', INSIGHT_PROMPT),
            ('human', f"""
                Question: {query}.
                Documentation: {context}
                """)
        ]
    )
    chain = prompt | llm | StrOutputParser()
    try:
        response = await chain.ainvoke({})
        return {
            "rag_insight": response
        }
    except:
        return {
            "rag_insight": None
        }