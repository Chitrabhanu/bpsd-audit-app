import pandas as pd
import streamlit as st

from config_loader import load_config
from connectors.factory import get_connector
from core.llm import make_llm_client
from core.audit import run_symptom_audit

config = load_config()

spark = None
if config["data"]["connector"] == "databricks":
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

connector = get_connector(config, spark=spark)
client = make_llm_client(config)
model = config["llm"]["model"]

st.title("BPSD Concept Evidence Dashboard")

st.caption(f"Running with environment: {config['env']}")

note_id = st.text_input("ClinicalNoteKey", value="48910906")

symptom_options = connector.get_symptom_categories()

selected_symptoms = st.multiselect(
    "Symptoms",
    options=symptom_options,
    default=symptom_options
)

if st.button("Run evidence audit"):
    if not note_id:
        st.error("Please enter a ClinicalNoteKey.")
        st.stop()

    if not selected_symptoms:
        st.error("Please select at least one symptom.")
        st.stop()

    note_units_df = connector.fetch_note_units(note_id)

    results = []

    progress = st.progress(0)

    for i, symptom_key in enumerate(selected_symptoms, start=1):
        st.write(f"Running {i}/{len(selected_symptoms)}: **{symptom_key}**")

        result = run_symptom_audit(
            connector=connector,
            client=client,
            model=model,
            note_units_df=note_units_df,
            symptom_key=symptom_key,
            retrieval_config=config["retrieval"]
        )

        result["note_id"] = str(note_id)
        results.append(result)

        progress.progress(i / len(selected_symptoms))

    results_df = pd.DataFrame(results)

    st.subheader("Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Concepts evaluated", len(results_df))
    c2.metric("Evidence present", int((results_df["evidence_present"] == "yes").sum()))
    c3.metric("Evidence absent", int((results_df["evidence_present"] == "no").sum()))
    c4.metric("Errors", int((results_df["evidence_present"] == "error").sum()))

    st.subheader("Evidence Summary")

    for _, row in results_df.iterrows():
        with st.container(border=True):
            st.markdown(f"### {row['target']}")
            st.write(f"Evidence: **{row['evidence_present']}**")
            st.write(f"Historical: **{row.get('historical', '')}**")
            st.write(f"Negated: **{row.get('negated', '')}**")
            st.code(row.get("evidence_excerpt", "") or "[none]")

    st.download_button(
        "Download results CSV",
        data=results_df.to_csv(index=False),
        file_name=f"bpsd_audit_{note_id}.csv",
        mime="text/csv"
    )