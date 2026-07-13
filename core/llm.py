import os
import json
import re
from openai import OpenAI

def extract_json_object(text):
    text = text.strip()
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)

    if not match:
        raise ValueError(f"No JSON object found in model output:\n{text}")

    return json.loads(match.group(0))


def build_prompt(input_key, evidence_text):
    return f"""
You are reviewing clinical note excerpts to determine whether there is evidence of the target symptom/domain.

Target symptom/domain: {input_key}

Respond strictly as valid JSON.

Required JSON schema:
{{
  "target": "{input_key}",
  "evidence_present": "yes/no",
  "historical": "yes/no",
  "negated": "yes/no",
  "evidence_excerpt": "exact quote from excerpts or empty string"
}}

Clinical note excerpts:
\"\"\"
{evidence_text}
\"\"\"
""".strip()


def make_llm_client(config):
    llm_config = config["llm"]
    provider = llm_config["provider"]

    if provider == "openai":
        return OpenAI(api_key=os.environ[llm_config["api_key_env"]])

    if provider == "databricks":
        token = os.environ["DATABRICKS_TOKEN"]
        return OpenAI(
            api_key=token,
            base_url=llm_config["base_url"]
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


def call_llm_for_symptom(client, model, symptom_key, evidence_text):
    prompt = build_prompt(symptom_key, evidence_text)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    answer = response.choices[0].message.content

    try:
        return extract_json_object(answer)
    except Exception as e:
        return {
            "target": symptom_key,
            "evidence_present": "error",
            "historical": "error",
            "negated": "error",
            "evidence_excerpt": "",
            "error": str(e),
            "raw_answer": answer
        }