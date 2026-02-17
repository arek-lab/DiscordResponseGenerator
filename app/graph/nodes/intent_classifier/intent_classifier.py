from langchain_core.prompts import ChatPromptTemplate

from app.graph.nodes.intent_classifier.prompt import INTENT_CLASSIFIER_PROMPT
from app.graph.nodes.models import IntentClassification
from app.graph.state import State
from config import get_openai


llm = get_openai().with_structured_output(IntentClassification)


async def intent_classifier(state: State) -> State:
    post= state["message"]['message']
    prompt = ChatPromptTemplate.from_messages(
        [("system", INTENT_CLASSIFIER_PROMPT), ("human", f"Post:\n{post}")], template_format="mustache"
    )
    chain = prompt | llm

    try:
        response: IntentClassification = await chain.ainvoke({})
        return {"intent" : response.intent}

    except Exception as e:
        return {"category": "Intent inference error"}
