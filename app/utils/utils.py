import ast
import re


def extract_xml_tags(xml_tags: str, xml_string: str) -> str:
	"""
	Extracts the xml string from tag using regular expression pattern.

	Args:
	----
	     xml_tags(str): The xml element to match.
	     xml_string(str): The string to match against.

	Returns:
	-------
	     The first match.

	"""
	pattern = f"<{xml_tags}.*?>(.*?)</{xml_tags}>"
	# Find all matches of the pattern in the XML string
	matches = re.findall(pattern, xml_string, re.DOTALL)
	return matches[0].replace("\n", "") if matches else ""


def convert_dict_string_to_dict(input_str: str) -> dict:
	"""
	Convert a string representation of a dictionary into a Python dictionary.

	Args:
	----
	input_str (str): The string representation of the dictionary.

	Returns:
	-------
	dict: The dictionary object.

	Raises:
	------
	ValueError: If the input string is not a valid dictionary.

	"""
	try:
		return ast.literal_eval(input_str)
	except SyntaxError as exc:
		raise ValueError("Input string is not a valid dictionary") from exc
