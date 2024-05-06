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


def extract_healthcare_information_from_json(json_query: dict, max_results: int = 5) -> str:
	"""
	Extracts healthcare information based on a JSON query.

	Returns a failure message if the query is too broad or if requested attributes are not present.
	If extraction exceeds the maximum allowed results, returns a truncated list.

	Args:
	----
	json_query (dict): Dictionary representing query attributes and values.
	max_results (int, optional): Maximum number of results to return. Defaults to 5.

	Returns:
	-------
	A string containing the extracted healthcare information.

	"""
	converted_json_query = {
		json_column_name_conversion[key]: value for key, value in json_query.items()
	}
	if all(value == "True" for value in converted_json_query.values()):
		attribute_conversion = [json_column_name_conversion[key] for key in converted_json_query]
		return f"""<failed_extraction>The extraction of the following attributes is impossible: {attribute_conversion}.</failed_extraction>
        Reason: the request is too broad and, as an agent, I cannot extract the entire dataset.

        Now, you must ask the user to be more specific about at least one of these attributes.
        You cannot use another tool.
        """

	columns_to_return = [key for key, value in converted_json_query.items() if value == "True"]
	columns_to_extract = [key for key, value in converted_json_query.items() if value != "True"]

	list_extracted_mask = [
		healthcare_dataset[column] == converted_json_query[column] for column in columns_to_extract
	]
	extracted_dataset = healthcare_dataset[np.logical_and.reduce(list_extracted_mask)][
		columns_to_extract + columns_to_return
	]

	if len(extracted_dataset) == 0:
		return f"""<failed_extraction>The extraction of the following attributes is impossible: {columns_to_extract}.</failed_extraction>
	Reason: the requested element are not present in the healthcare dataset.

	Now, you must ask the user to double-check and provide a correct answer.
	You cannot use another tool.
	"""

	output_string = ""
	if len(extracted_dataset) > max_results:
		extracted_dataset = extracted_dataset.head(max_results)
		output_string = f"The extraction is too big. Here are {max_results} truncated results:"
	output_string += "<healthcare_dataset>"
	for _, row in extracted_dataset.iterrows():
		output_string += "\n" + str(dict(row))
	output_string += "</healthcare_dataset>"

	output_string += """Now, provide a concise answer to the user.
	You cannot use another tool.
	"""
	return output_string
