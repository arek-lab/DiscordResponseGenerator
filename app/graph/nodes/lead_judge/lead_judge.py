from langchain_core.prompts import ChatPromptTemplate

from config import get_anthropic
from app.graph.state import State
from app.graph.nodes.models import LeadJudgeModel
from app.graph.nodes.lead_judge.prompt import LEAD_JUDGE_PROMPT

llm = get_anthropic().with_structured_output(LeadJudgeModel)

async def lead_judge(state: State) -> State:
    post= state["message"]['message']
    intent = state["intent"]
    domain = state["domain"]
    prompt = ChatPromptTemplate.from_messages(
        [("system", LEAD_JUDGE_PROMPT), 
         ("human", 
          f"""Judge this lead:
          user_message: {post},
          intent: {intent},
          domain: {domain}
          """)]
    )
    chain = prompt | llm

    try:
        response: LeadJudgeModel = await chain.ainvoke({})
        return {
            "lead_judge": LeadJudgeModel(
            is_lead=response.is_lead,
            lead_score=response.lead_score,
            reason=response.reason,
            devdocs_query=response.devdocs_query,
            insight=response.insight
        )
        }

    except Exception as e:
        return {
            "lead_judge": LeadJudgeModel(
            is_lead=False,
            lead_score=0,
            reason="Lead Judge inference error",
            devdocs_query="Lead Judge inference error",
            insight="Lead Judge inference error"
        )
        }