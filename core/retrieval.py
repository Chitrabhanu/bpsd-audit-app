import re
import numpy as np
import pandas as pd


CATEGORY_KEYWORDS = {
    "dementia": ["dementia", "alzheimer", "neurocognitive disorder", "memory impairment", "cognitive decline"],
    "agitation": ["agitation", "agitated", "restless", "pacing", "behavioral disturbance"],
    "aggression": ["aggression", "aggressive", "combative", "hitting", "kicking", "threatening"],
    "anxiety": ["anxiety", "anxious", "worry", "worried", "fearful", "panic"],
    "apathy": ["apathy", "apathetic", "lack of motivation", "withdrawn", "decreased initiative"],
    "delusion": ["delusion", "delusions", "paranoid", "false belief", "suspiciousness", "stealing"],
    "depression": ["depression", "depressed", "sadness", "tearful", "hopeless", "loss of pleasure"],
    "disinhibition": ["disinhibition", "disinhibited", "inappropriate behavior", "impulsive", "sexual behavior", "poor impulse control"],
    "hallucination": ["hallucination", "hallucinations", "visual hallucination", "auditory hallucination", "seeing people", "hearing voices"],
    "irritability": ["irritability", "irritable", "irritated", "anger", "angry", "short temper", "short-tempered", "frustrated", "easily frustrated", "outburst", "outbursts"],
    "motor_disturbance": ["motor disturbance", "wandering", "pacing", "repetitive movement", "restlessness", "purposeless activity"],
    "sleep_disturbance": ["sleep disturbance", "insomnia", "difficulty sleeping", "poor sleep", "frequent awakenings"],
    "nocturnal_activities": ["nocturnal", "awake at night", "wandering at night", "sundowning", "nighttime confusion"],
    "delirium": ["delirium", "acute confusion", "altered mental status", "fluctuating attention", "encephalopathy"],
    "mci": ["mild cognitive impairment", "mci", "mild cognitive decline", "amnestic mci"],
    "cognitive_impairment": ["cognitive impairment", "impaired cognition", "memory impairment", "executive dysfunction", "cognitive decline"],
    "neurodegeneration": ["neurodegeneration", "neurodegenerative", "brain atrophy", "cortical atrophy", "progressive neurologic decline"],
}


def cosine_similarity(a, b):
    """
    Compute cosine similarity between two embedding vectors.
    """

    if a is None or b is None:
        return None

    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    if a.size == 0 or b.size == 0:
        return None

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return None

    return float(np.dot(a, b) / denom)


def keyword_hit(unit_text, keywords):
    """
    Return keyword hits found in a unit of clinical text.
    """

    text = str(unit_text).lower()
    hits = []

    for kw in keywords:
        kw_lower = kw.lower()

        if " " in kw_lower:
            if kw_lower in text:
                hits.append(kw)
        else:
            pattern = r"\b" + re.escape(kw_lower) + r"\b"

            if re.search(pattern, text):
                hits.append(kw)

    return hits


def retrieve_evidence_for_symptom(
    note_units_pdf,
    symptom_terms_pdf,
    symptom_key,
    top_k_per_query=5,
    max_semantic_units=12,
    max_final_units=20,
    min_similarity=0.55,
):
    """
    Retrieve candidate evidence excerpts for one symptom.

    This combines:
    1. embedding similarity retrieval
    2. keyword retrieval
    3. deduplication by UnitRowId
    4. final ordering by UnitId

    Returns:
        evidence_text: formatted text block passed to the LLM
        final_candidates: dataframe of retrieved evidence units
    """

    semantic_matches = []

    # ------------------------------------------------------------
    # Semantic retrieval using cosine similarity
    # ------------------------------------------------------------

    for _, unit in note_units_pdf.iterrows():
        unit_emb = unit.get("UnitTextEmbedding")

        for _, query in symptom_terms_pdf.iterrows():
            query_emb = query.get("QueryTextEmbedding")

            sim = cosine_similarity(unit_emb, query_emb)

            if sim is None:
                continue

            semantic_matches.append({
                "UnitRowId": unit.get("UnitRowId"),
                "ClinicalNoteKey": unit.get("ClinicalNoteKey"),
                "EncounterKey": unit.get("EncounterKey"),
                "PatientDurableKey": unit.get("PatientDurableKey"),
                "Section": unit.get("Section"),
                "UnitId": int(unit.get("UnitId")),
                "UnitText": unit.get("UnitText"),
                "RetrievalMethod": "embedding",
                "Category": symptom_key,
                "MatchedQuery": query.get("QueryText"),
                "Similarity": sim,
                "KeywordHits": [],
            })

    semantic_pdf = pd.DataFrame(semantic_matches)

    if semantic_pdf.empty:
        semantic_selected = pd.DataFrame()
    else:
        semantic_selected = (
            semantic_pdf
            .sort_values(["MatchedQuery", "Similarity"], ascending=[True, False])
            .groupby("MatchedQuery", as_index=False)
            .head(top_k_per_query)
        )

        semantic_selected = semantic_selected[
            semantic_selected["Similarity"] >= min_similarity
        ]

        semantic_selected = (
            semantic_selected
            .sort_values("Similarity", ascending=False)
            .drop_duplicates(subset=["UnitRowId"])
            .head(max_semantic_units)
        )

    # ------------------------------------------------------------
    # Keyword retrieval
    # ------------------------------------------------------------

    keywords = CATEGORY_KEYWORDS.get(symptom_key, [])
    keyword_rows = []

    for _, unit in note_units_pdf.iterrows():
        hits = keyword_hit(unit.get("UnitText"), keywords)

        if hits:
            keyword_rows.append({
                "UnitRowId": unit.get("UnitRowId"),
                "ClinicalNoteKey": unit.get("ClinicalNoteKey"),
                "EncounterKey": unit.get("EncounterKey"),
                "PatientDurableKey": unit.get("PatientDurableKey"),
                "Section": unit.get("Section"),
                "UnitId": int(unit.get("UnitId")),
                "UnitText": unit.get("UnitText"),
                "RetrievalMethod": "keyword",
                "Category": symptom_key,
                "MatchedQuery": ", ".join(hits),
                "Similarity": None,
                "KeywordHits": hits,
            })

    keyword_selected = pd.DataFrame(keyword_rows)

    # ------------------------------------------------------------
    # Combine semantic + keyword results
    # ------------------------------------------------------------

    frames = []

    if not semantic_selected.empty:
        frames.append(semantic_selected)

    if not keyword_selected.empty:
        frames.append(keyword_selected)

    if not frames:
        return "", pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    final_candidates = (
        combined
        .groupby(
            [
                "UnitRowId",
                "ClinicalNoteKey",
                "EncounterKey",
                "PatientDurableKey",
                "Section",
                "UnitId",
                "UnitText",
                "Category",
            ],
            as_index=False,
            dropna=False,
        )
        .agg({
            "RetrievalMethod": lambda x: "+".join(sorted(set(x))),
            "MatchedQuery": lambda x: " | ".join(
                sorted(set(str(v) for v in x if pd.notnull(v)))
            ),
            "Similarity": "max",
            "KeywordHits": lambda x: sorted(set(
                hit
                for hits in x
                for hit in (hits if isinstance(hits, list) else [])
            )),
        })
        .sort_values("UnitId")
        .head(max_final_units)
    )

    # ------------------------------------------------------------
    # Build LLM evidence text
    # ------------------------------------------------------------

    evidence_lines = []

    for _, row in final_candidates.iterrows():
        evidence_lines.append(
            f"[UnitId={row['UnitId']} | Section={row['Section']}]\n"
            f"{row['UnitText']}"
        )

    evidence_text = "\n\n".join(evidence_lines)

    return evidence_text, final_candidates