import numpy as np
import pandas as pd

from app.shared.global_settings import BASEDIR

healthcare_dataset = pd.read_csv(f"{BASEDIR}/data/healthcare_dataset.csv")

json_column_name_conversion = {
    "patient_name": "Name",
    "patient_blood_type": "Blood Type",
    "patient_doctor": "Doctor",
    "patient_hospital": "Hospital",
    "patient_medical_condition": "Medical Condition",
}


def extract_healthcare_information_from_json(json_query: dict) -> str:
    converted_json_query = {
        json_column_name_conversion[key]: value for key, value in json_query.items()
    }
    if all(value == "True" for value in converted_json_query.values()):
        attribute_conversion = [
            json_column_name_conversion[key] for key in converted_json_query.keys()
        ]
        output_string = (
            """<failed_extraction>The extraction of the following attributes is impossible: %s.</failed_extraction>
        Reason: the request is too broad and, as an agent, I cannot extract the entire dataset.

        Now, you must ask the user to be more specific about at least one of these attributes.
        You cannot use another tool.
        """
            % attribute_conversion
        )
        return output_string

    else:
        columns_to_return = [
            key for key, value in converted_json_query.items() if value == "True"
        ]
        columns_to_extract = [
            key for key, value in converted_json_query.items() if value != "True"
        ]

        list_extracted_mask = []
        for column in columns_to_extract:
            list_extracted_mask.append(
                healthcare_dataset[column] == converted_json_query[column]
            )
        extracted_dataset = healthcare_dataset[
            np.logical_and.reduce(list_extracted_mask)
        ][columns_to_extract + columns_to_return]

        if len(extracted_dataset) == 0:
            output_string = (
                """<failed_extraction>The extraction of the following attributes is impossible: %s.</failed_extraction>
        Reason: the requested element are not present in the healthcare dataset.

        Now, you must ask the user to double-check and provide a correct answer.
        You cannot use another tool.
        """
                % columns_to_extract
            )
            return output_string
        else:
            output_string = ""
            if len(extracted_dataset) > 5:
                extracted_dataset = extracted_dataset.head(5)
                output_string = (
                    "The extraction is too big. Here are 5 truncated results:"
                )
            output_string += "<healthcare_dataset>"
            for _, row in extracted_dataset.iterrows():
                output_string += "\n" + str(dict(row))
            output_string += "</healthcare_dataset>"

            output_string += """Now, provide a concise answer to the user.
            You cannot use another tool.
            """
            return output_string
