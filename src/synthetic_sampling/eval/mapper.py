import re
import pandas as pd
from typing import Dict, Any

class Mapper:
    def __init__(self, survey_mappings: Dict[str, Dict[str, Any]]):
        self.survey_mappings = survey_mappings
        # create reverse mapping from feature names to their sections for quick lookup
        self.feature_to_section = {
            feature: section
            for section, features in self.survey_mappings.items()
            for feature in features
        }

    def map_value(self, feature_name: str, value):
        section = self.feature_to_section.get(feature_name)
        if not section:
            return str(value) 

        feature_mapping = self.survey_mappings[section].get(feature_name)
        if not feature_mapping:
            return str(value)  

        values_mapping = feature_mapping.get('values', {})
        if pd.isnull(value):
            return "Missing"

        if isinstance(value, float) and value.is_integer():
            value_key = str(int(value))
        elif isinstance(value, (int, np.integer)):
            value_key = str(value)
        else:
            value_key = str(value)

        return values_mapping.get(value_key, str(value))

    def fill_prompt(self, respondent: pd.Series, prompt_template: str):
        placeholders = {}
        placeholder_pattern = re.compile(r"\{(\w+)\}")
        placeholder_names = placeholder_pattern.findall(prompt_template)

        for placeholder in placeholder_names:
            if placeholder in respondent:
                value = respondent[placeholder]
                if placeholder in ['agea']: # note to self: replace this line with code to handle any variable
                    # handle numeric fields separately
                    placeholders[placeholder] = str(int(value)) if pd.notnull(value) else f"unknown {placeholder}"
                else:
                    mapped_value = self.map_value(placeholder, value)
                    placeholders[placeholder] = mapped_value
            else:
                placeholders[placeholder] = "Unknown"

        filled_prompt = prompt_template.format(**placeholders)
        return filled_prompt

