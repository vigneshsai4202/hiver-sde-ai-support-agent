import pandas as pd

REQUIRED = ["human_intent","human_escalate",
            "human_resolution_note","annotator_notes"]

def validate_golden(path):
    df = pd.read_csv(path)
    errors = [f"Missing column: {c}" for c in REQUIRED if c not in df.columns]
    if "golden_id" in df and df["golden_id"].duplicated().any():
        errors.append("Duplicate golden_id")
    if "conversation_id" in df and df["conversation_id"].duplicated().any():
        errors.append("Repeated conversation_id")
    return errors
