from langgraph.graph import END

from app.graph.state import State

def technical_classification_gate(state: State) -> str:
    if state["category"] == "technical_problem":
        return "intent_classifier"
    return END

def intent_classification_gate(state: State) -> str:
    if state["intent"] != "out_of_scope":
        return "domain_classifier"
    return END
def domain_classification_gate(state: State) -> str:
    if state["intent"] != "out_of_scope":
        return "lead_judge"
    return END
def lead_judge_gate(state: State) -> str:
    if state["lead_judge"].is_lead == True:
        return "generate_response"
    return "process_rag"


def should_regenerate(state: State) -> str:
    """
    Routing logic after validation.
    """
    validation = state["validation"]
    regenerations_attempt = state.get("regenerations_attempt", 0)
    print(regenerations_attempt, validation.decision)
    MAX_ATTEMPTS = 2

    if validation.decision != "approve" and regenerations_attempt < MAX_ATTEMPTS:
        return "dynamic_prompt"

    return END

def should_reply(state: State) -> str:
    c = state["classification"]
    if c.category in ["support", "already_resolved", "offtopic"]:
        return END

    if c.category.category == "too_vague":
        return END

    if c.category.category == "technical_help":
        return "generate_response"

    return END