from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import State
from app.graph.nodes.models import ReplyModel
from config import get_openai
from app.graph.nodes.reputation_response.prompt import GENERATE_REPUTATION_REPLY

llm = get_openai().with_structured_output(ReplyModel)

async def reputation_response(state: State) -> State:
    original_message = state["message"]["message"]
    domain = state["domain"]
    intent = state["intent"]
    lead_score = state["lead_judge"].lead_score
    insight = state["rag_insight"] or None

    prompt = ChatPromptTemplate.from_messages([
        ("system", GENERATE_REPUTATION_REPLY),
        ("human", 
          f'''Generate reply:
          Original message: {original_message}.
          Domain: {domain}.
          Intent: {intent}.
          Lead_score: {lead_score}.
          Insight: {insight}.
          ''')])
    
    chain = prompt | llm

    try:
        response: ReplyModel = await chain.ainvoke({})
        return {"reply": ReplyModel(
            reply=response.reply,
            tone=response.tone,
            cta_type=response.cta_type,
        )}
    except:
        {"reply": ReplyModel(
            reply="Response generation error",
            tone="Response generation error",
            cta_type="Response generation error",
        )}