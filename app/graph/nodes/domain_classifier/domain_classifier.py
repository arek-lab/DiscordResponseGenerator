from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.intent_classifier.prompt import INTENT_CLASSIFIER_PROMPT
from app.graph.nodes.models import DomainClassification
from app.graph.state import State
from config import get_openai


llm = get_openai().with_structured_output(DomainClassification)


async def domain_classifier(state: State) -> State:
    post= state["message"]['message']
    prompt = ChatPromptTemplate.from_messages(
        [("system", INTENT_CLASSIFIER_PROMPT), ("human", f"Post:\n{post}")], template_format="mustache"
    )
    chain = prompt | llm

    try:
        response: DomainClassification = await chain.ainvoke({})
        return {"domain": response.domain}

    except Exception as e:
        # Fallback - treat as too_vague
        print(e)
        return {"category": "Category inference error"}
