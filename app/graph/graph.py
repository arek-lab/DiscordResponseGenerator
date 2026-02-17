from langgraph.graph import END, START, StateGraph

from app.graph.state import State
from app.graph.contitional_edges import *
from app.graph.nodes.techical_classifier.techical_classifier import techical_classifier
from app.graph.nodes.intent_classifier.intent_classifier import intent_classifier
from app.graph.nodes.domain_classifier.domain_classifier import domain_classifier
from app.graph.nodes.lead_judge.lead_judge import lead_judge
from app.graph.nodes.lead_reposnse.generate_response import generate_response
from app.graph.nodes.process_rag.process_rag import process_rag
from app.graph.nodes.reputation_response.reputation_response import reputation_response

flow = StateGraph(State)


flow.add_node("techical_classifier", techical_classifier)
flow.add_node("intent_classifier", intent_classifier)
flow.add_node("domain_classifier", domain_classifier)
flow.add_node("lead_judge", lead_judge)
flow.add_node("generate_response", generate_response)
flow.add_node("process_rag", process_rag)
flow.add_node("reputation_response", reputation_response)


flow.add_edge(START, "techical_classifier")
flow.add_conditional_edges("techical_classifier", technical_classification_gate, {
    "intent_classifier":"intent_classifier",
    END: END
})
flow.add_conditional_edges("intent_classifier", intent_classification_gate, {
    "domain_classifier":"domain_classifier",
    END: END
})

flow.add_conditional_edges("domain_classifier", domain_classification_gate, {
    "lead_judge":"lead_judge",
    END: END
})

flow.add_conditional_edges("lead_judge", lead_judge_gate, {
    "generate_response":"generate_response",
    "process_rag":"process_rag"
})
flow.add_edge("process_rag", "reputation_response")



graph = flow.compile()

graph.get_graph().draw_mermaid_png(output_file_path="graph.png")
