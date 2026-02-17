from typing import List, Literal, Optional

from pydantic import BaseModel, Field

class TechnicalClassification(BaseModel):
    category: Literal["technical_problem", "not_technical"]

class IntentClassification(BaseModel):
    intent: Literal["debugging", "planning", "migration", "optimization", "integration", "evaluation", "scaling", "out_of_scope"]

class DomainClassification(BaseModel):
    domain: Literal["database", "auth", "api_integration", "deployment", "scaling", "security", "migration", "mcp", "commercialization", "architecture", "out_of_scope"]

class LeadJudgeModel(BaseModel):
    is_lead: bool
    lead_score: float
    reason: str | None
    devdocs_query: str | None = Field(description="populated only when is_lead=False")
    insight: str | None = Field(description="populated only when is_lead=True")

class ReplyModel(BaseModel):
    reply: str = Field(description="Ready to use the Discord reply message text")
    tone: Literal["peer", "helpful", "technical", "Response generation error"]
    cta_type: Literal["dm_invite", "offer_help", "share_experience", "Response generation error"] 
