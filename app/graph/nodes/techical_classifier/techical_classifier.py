from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.techical_classifier.prompt import TECHNICAL_CLASSIFIER_PROMPT
from app.graph.nodes.models import TechnicalClassification
from app.graph.state import State
from config import get_openai

llm = get_openai().with_structured_output(TechnicalClassification)


async def techical_classifier(state: State) -> State:
    post= state["message"]['message']
    prompt = ChatPromptTemplate.from_messages(
        [("system", TECHNICAL_CLASSIFIER_PROMPT), ("human", f"Post:\n{post}")], template_format="mustache"
    )
    chain = prompt | llm

    try:
        response: TechnicalClassification = await chain.ainvoke({})
        return {"category": response.category}

    except Exception as e:
        # Fallback
        return {"category": "Category inference error"}
