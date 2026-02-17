from langchain_core.prompts import ChatPromptTemplate

from app.graph.state import State
from app.graph.nodes.models import ReplyModel
from config import get_openai
from app.graph.nodes.lead_reposnse.prompt import GENERATE_RESPONSE_POST


llm = get_openai().with_structured_output(ReplyModel)


async def generate_response(state: State) -> State:

    original_message = state["message"]["message"]
    domain = state["domain"]
    intent = state["intent"]
    lead_score = state["lead_judge"].lead_score
    insight = state["lead_judge"].insight

    prompt = ChatPromptTemplate.from_messages(
        [("system", GENERATE_RESPONSE_POST),
         ("human", 
          f'''Generate reply:
          Original message: {original_message}.
          Domain: {domain}.
          Intent: {intent}.
          Lead_score: {lead_score}.
          Insight: {insight}.
          ''')],
        template_format="mustache",
    )

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
