import pandas as pd
from typing import Dict, Any, List

def apply_mapping(df: pd.DataFrame, mapping: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Applies numeric-to-text mapping to the DataFrame based on survey_mappings.

    Args:
        df: The original survey DataFrame.
        mapping: Nested dictionary containing survey sections and question mappings.

    Returns:
        The mapped DataFrame where numeric values are replaced with text equivalents.
    """
    for category, variables in mapping.items():
        for col, details in variables.items():
            if col in df.columns:
                value_mapping = details.get("values", {})
                # replace based on mapping; keep original value if not mapped
                df[col] = df[col].astype(str).map(value_mapping).fillna(df[col].astype(str))
    return df

def replace_special_codes(series: pd.Series, code_map: Dict[str, Optional[str]]) -> pd.Series:
    """
    Replaces or excludes special codes (various missings from original survey data) in a df column based on a mapping.

    Args:
        series: df column as pd.Series.
        code_map: A dictionary mapping codes to their replacements.
                  If a code maps to `None`, rows with that code are excluded.

    Returns:
        pd.Series with replacements (specified codes excluded).
    """
    drop_codes = [code for code, replacement in code_map.items() if replacement is None]
    drop_mask = series.isin(drop_codes)

    replaced_series = series.replace({code: replacement for code, replacement in code_map.items() if replacement is not None})
    replaced_series[drop_mask] = pd.NA

    return replaced_series

def get_question_text_from_mapping(survey_mappings: Dict[str, Dict[str, Any]], question_id: str) -> str:
    """
    Retrievs the question text for a given question ID from the survey mappings.

    Args:
        survey_mappings: Nested dictionary containing survey sections and questions.
        question_id: The question ID to retrieve text for.

    Returns:
        The question text if found, else a default message.
    """
    for category, variables in survey_mappings.items():
        if question_id in variables:
            q_info = variables[question_id]
            return q_info.get("question", f"Question text not found for {question_id}")
    return f"No text found for {question_id}"

def build_prompt_data(
    df: pd.DataFrame,
    qid: str,
    mapper,
    config: Any,  # Replace with EvaluationConfig if importing
    survey_mappings: Dict[str, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Prepares prompts for each respondent in df for evaluation per given question.

    Args:
        df: The survey DataFrame.
        qid: The question ID (column name) to evaluate.
        mapper: An instance of the Mapper class for value mapping.
        config: The EvaluationConfig instance containing configuration parameters.
        survey_mappings: Nested dictionary containing survey sections and questions.

    Returns:
        A list of dictionaries, each containing:
            - 'prompt': The full prompt string for the respondent.
            - 'label_options': List of possible answer labels.
            - 'true_label': The respondent's actual answer.
    """
    if qid not in df.columns:
        print(f"Warning: {qid} not in DataFrame columns. Skipping.")
        return []

    # 1. Replace special codes
    col_series = replace_special_codes(df[qid], config.special_codes)

    # 2. Filter out rows that are now NA
    valid_mask = col_series.notna()
    sub_df = df[valid_mask].copy()
    sub_df[qid] = col_series[valid_mask]

    if sub_df.empty:
        print(f"No valid responses for {qid}. Skipping.")
        return []

    # 3. Gather label options
    label_options = sorted(sub_df[qid].unique().tolist())

    # 4. Retrieve question text
    question_text = get_question_text_from_mapping(survey_mappings, qid)
    question_prompt = f"\nPlease answer the following question:\n{question_text}\n"

    # 5. Build prompts
    results = []
    system_content = (
        "You are a helpful AI assistant for public opinion research. "
        "You are skillful at using your knowledge to make good judgment "
        "about people's preferences when given some background information.\n"
    )

    for _, row in sub_df.iterrows():
        filled_profile = mapper.fill_prompt(row, config.profile_prompt_template)
        user_content = filled_profile + question_prompt

        final_prompt = f"system: {system_content}user: {user_content}\n"

        results.append({
            "prompt": final_prompt,
            "label_options": label_options,
            "true_label": row[qid]
        })

    return results

