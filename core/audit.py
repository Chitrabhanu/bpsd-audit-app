from core.retrieval import retrieve_evidence_for_symptom
from core.llm import call_llm_for_symptom

def run_symptom_audit(connector, client, model, note_units_df, symptom_key, retrieval_config):
    symptom_terms_df = connector.fetch_symptom_terms(symptom_key)

    if note_units_df.empty:
        return {
            "target": symptom_key,
            "evidence_present": "error",
            "historical": "error",
            "negated": "error",
            "evidence_excerpt": "",
            "error": "No note embeddings found."
        }

    if symptom_terms_df.empty:
        return {
            "target": symptom_key,
            "evidence_present": "error",
            "historical": "error",
            "negated": "error",
            "evidence_excerpt": "",
            "error": f"No symptom embeddings found for {symptom_key}."
        }

    evidence_text, _ = retrieve_evidence_for_symptom(
        note_units_pdf=note_units_df,
        symptom_terms_pdf=symptom_terms_df,
        symptom_key=symptom_key,
        top_k_per_query=retrieval_config["top_k_per_query"],
        max_semantic_units=retrieval_config["max_semantic_units"],
        max_final_units=retrieval_config["max_final_units"],
        min_similarity=retrieval_config["min_similarity"],
    )

    if not evidence_text.strip():
        return {
            "target": symptom_key,
            "evidence_present": "no",
            "historical": "no",
            "negated": "no",
            "evidence_excerpt": ""
        }

    return call_llm_for_symptom(
        client=client,
        model=model,
        symptom_key=symptom_key,
        evidence_text=evidence_text
    )