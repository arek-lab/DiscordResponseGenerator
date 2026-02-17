from typing import Annotated, TypedDict

from app.graph.nodes.models import *


class State(TypedDict):
    message: Annotated[str, "Original messsge"]
    category: TechnicalClassification 
    intent: IntentClassification
    domain: DomainClassification
    lead_judge: LeadJudgeModel
    reply: ReplyModel
    rag_insight: str
    
