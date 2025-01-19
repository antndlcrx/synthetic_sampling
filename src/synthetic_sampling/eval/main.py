import os
import pandas as pd

from config import EvaluationConfig
from mapper import Mapper
from data_processing import apply_mapping, build_prompt_data
from model_inference import compute_label_logprobs_batch_mc
from metrics import compute_instance_metrics

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)


#################################
# 1) load model and tokenizer
#################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "oxford-llms/llama3-1-ox-llms-8b-sft-full"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN,
                                          padding_side='left')
tokenizer.pad_token = tokenizer.eos_token


model = AutoModelForCausalLM.from_pretrained(model_name,
                                             trust_remote_code=True,
                                             torch_dtype=torch.float16,
                                             device_map='auto',
                                             token=HF_TOKEN,
                                             )


#################################
# 2) load survey and mappings
#################################

directory = "profiles/mappings/WVS"
survey_mappings = {}

with open(directory, 'r', encoding='utf-8') as file:
    survey_mappings = json.load(file)

wvs = pd.read_csv("GDRIVE_LINK/WVS_2017_22.csv")

# get test ids
wvs_dataset = load_dataset("oxford-llms/world_values_survey_2017_2022_sft")
test_ids = wvs_dataset['test']['id']
wvs_test = wvs[wvs['D_INTERVIEW'].isin(test_ids)]

wvs_test_mapped = apply_mapping(wvs_test, survey_mappings)

#################################
# 3) set eval config
#################################

question_ids = ["Q22", "Q25", "Q17", "Q18", "Q206"] # which questions to evaluate

profile_prompt_template = (
    "Imagine you are a {Q262}-year old {Q260} living in {B_COUNTRY}. "
    "Your highest education is {Q275}. "
) # var names to be changed depending on survey

special_codes = {
        "No answer": None,
        "Not asked": None,
        "Missing; Not available": None,
        "Don´t know": 'Don´t know',
        "Don't know": 'Don´t know',
        "-5": None
    }

config = EvaluationConfig(
    question_ids=question_ids,
    special_codes=special_codes,
    profile_prompt_template=profile_prompt_template,
    model=model,
    tokenizer=tokenizer,
    device=device,
    batch_size=6,
    f1_average="weighted"
)

#################################
# 4) implement pipeline
#################################

def evaluate_questions_pipeline(
    df: pd.DataFrame,
    survey_mappings: Dict[str, Dict[str, Any]],
    mapper,
    config: EvaluationConfig
) -> pd.DataFrame:
    """

    Runs the full eval pipeline:

      - Loops over question_ids in config
      - For each qid, builds prompt data
      - Batches model inference with multiple-choice approach
      - Computes metrics
      - Returns a results DataFrame.

    Args:
        df: The survey DataFrame.
        survey_mappings: Nested dictionary containing survey sections and questions.
        mapper: An instance of the Mapper class for value mapping.
        config: The EvaluationConfig instance containing configuration parameters.

    Returns:
        A pd.DataFrame with columns:
            [
              "question_id",
              "question_text",
              "NLL",
              "Brier",
              "F1",
              "n_observations",
              "n_labels",
              "labels",
              "prompt_example",
            ]
    """
    results = []

    for qid in config.question_ids:
        # 1. Build the data items for this question
        data_items = build_prompt_data(df, qid, mapper, config, survey_mappings)
        if not data_items:
            continue  # Skip if no data

        # 2. Compute logprobs
        distributions = compute_label_logprobs_batch_mc(
            model=config.model,
            tokenizer=config.tokenizer,
            data_list=data_items,
            device=config.device,
            batch_size=config.batch_size
        )

        # 3. Compute metrics
        true_labels = [item["true_label"] for item in data_items]
        label_options = data_items[0]["label_options"]  
        avg_nll, avg_brier, f1_val = compute_instance_metrics(
            distributions=distributions,
            true_labels=true_labels,
            label_options=label_options,
            f1_average=config.f1_average
        )

        # 4. store results
        record = {
            "question_id": qid,
            "question_text": get_question_text_from_mapping(survey_mappings, qid),
            "NLL": avg_nll,
            "Brier": avg_brier,
            "F1": f1_val,
            "n_observations": len(data_items),
            "n_labels": len(label_options),
            "labels": label_options,
            "prompt_example": data_items[0]["prompt"][:300] if data_items else ""
        }
        results.append(record)

    return pd.DataFrame(results)


#################################
# 4) run pipeline store results
#################################

results_df = evaluate_questions_pipeline(
    df=wvs_sample_mapped,
    survey_mappings=survey_mappings,
    mapper=mapper,
    config=config
)


results_df.to_csv("results/eval_results.csv")
