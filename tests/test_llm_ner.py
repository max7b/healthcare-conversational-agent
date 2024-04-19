from app.src.llm_ner_to_json import format_query_to_json_with_llm_for_ner
from app.src.utils import extract_xml_tags, string_to_dict


# Test 1 - Retrieve patient Samantha Rodriguez's blood type and medical condition
def test_retrieve_samantha_rodriguez_info():
    query = "What is the blood type and medical condition of Samantha Rodriguez?"
    expected_json = {
        "patient_name": "Samantha Rodriguez",
        "patient_blood_type": "True",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 2 - Retrieve patient Michael Johnson's doctor
def test_retrieve_michael_johnson_doctor():
    query = "Who is the doctor for patient Michael Johnson?"
    expected_json = {"patient_name": "Michael Johnson", "patient_doctor": "True"}
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 3 - Retrieve patient Tiffany Davis's hospital information
def test_retrieve_tiffany_davis_hospital():
    query = "Please provide the hospital information for patient Tiffany Davis."
    expected_json = {"patient_name": "Tiffany Davis", "patient_hospital": "True"}
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 4 - Retrieve patient Ethan Nguyen's medical condition
def test_retrieve_ethan_nguyen_medical_condition():
    query = "I need to know the medical condition of patient Ethan Nguyen."
    expected_json = {
        "patient_name": "Ethan Nguyen",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 5 - Retrieve patient Jessica Lee's blood type and doctor
def test_retrieve_jessica_lee_blood_type_and_doctor():
    query = "Can you tell me the blood type and doctor for patient Jessica Lee?"
    expected_json = {
        "patient_name": "Jessica Lee",
        "patient_blood_type": "True",
        "patient_doctor": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 6 - Retrieve patient David Patel's hospital and medical condition
def test_retrieve_david_patel_hospital_and_medical_condition():
    query = "What is the hospital and medical condition for patient David Patel?"
    expected_json = {
        "patient_name": "David Patel",
        "patient_hospital": "True",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 7 - Retrieve patient information for a patient with Hypertension
def test_retrieve_patient_info_for_hypertension():
    query = "I need the patient name, blood type, and doctor for a patient with Hypertension."
    expected_json = {
        "patient_name": "True",
        "patient_blood_type": "True",
        "patient_doctor": "True",
        "patient_medical_condition": "Hypertension",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 8 - Retrieve patient information for a patient with blood type B+
def test_retrieve_patient_info_for_blood_type_b_plus():
    query = "Can you provide the hospital and medical condition for a patient with blood type B+?"
    expected_json = {
        "patient_blood_type": "B+",
        "patient_hospital": "True",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 9 - Retrieve patient information for a patient with Arthritis
def test_retrieve_patient_info_for_arthritis():
    query = "I need the name, doctor, and hospital for a patient with Arthritis."
    expected_json = {
        "patient_name": "True",
        "patient_doctor": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "Arthritis",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 10 - Retrieve patient Olivia Gonzalez's blood type and medical condition
def test_retrieve_olivia_gonzalez_blood_type_and_medical_condition():
    query = "What is the blood type and medical condition for patient Olivia Gonzalez?"
    expected_json = {
        "patient_name": "Olivia Gonzalez",
        "patient_blood_type": "True",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 11 - Retrieve patient Ryan Nguyen's doctor and hospital
def test_retrieve_ryan_nguyen_doctor_and_hospital():
    query = "Can you tell me the doctor and hospital for patient Ryan Nguyen?"
    expected_json = {
        "patient_name": "Ryan Nguyen",
        "patient_doctor": "True",
        "patient_hospital": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 12 - Retrieve patient information for a patient at Walton LLC hospital
def test_retrieve_patient_info_for_walton_llc_hospital():
    query = "I need the patient name, blood type, and medical condition for a patient at Walton LLC hospital."
    expected_json = {
        "patient_name": "True",
        "patient_blood_type": "True",
        "patient_medical_condition": "True",
        "patient_hospital": "Walton LLC",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 13 - Retrieve patient information for a patient with Diabetes
def test_retrieve_patient_info_for_diabetes():
    query = "Please provide the doctor and hospital information for a patient with Diabetes."
    expected_json = {
        "patient_doctor": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "Diabetes",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 14 - Retrieve patient information for a patient treated by Dr. Diane Jackson
def test_retrieve_patient_info_for_dr_diane_jackson():
    query = "I need the name, blood type, and medical condition for a patient treated by Dr. Diane Jackson."
    expected_json = {
        "patient_name": "True",
        "patient_blood_type": "True",
        "patient_medical_condition": "True",
        "patient_doctor": "Diane Jackson",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 15 - Retrieve patient information for a patient with Asthma
def test_retrieve_patient_info_for_asthma():
    query = "Can you give me the patient name, doctor, and hospital for a patient with Asthma?"
    expected_json = {
        "patient_name": "True",
        "patient_doctor": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "Asthma",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 16 - Retrieve patient information for a patient at Brown Inc hospital
def test_retrieve_patient_info_for_brown_inc_hospital():
    query = "I need the blood type, doctor, and medical condition for a patient at Brown Inc hospital."
    expected_json = {
        "patient_blood_type": "True",
        "patient_doctor": "True",
        "patient_medical_condition": "True",
        "patient_hospital": "Brown Inc",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 17 - Retrieve patient information for a patient with blood type A-
def test_retrieve_patient_info_for_blood_type_a_minus():
    query = "What is the name, hospital, and medical condition for a patient with blood type A-?"
    expected_json = {
        "patient_name": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "True",
        "patient_blood_type": "A-",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 18 - Retrieve patient information for a patient with Obesity
def test_retrieve_patient_info_for_obesity():
    query = "Can you provide the patient name, doctor, and hospital for a patient with Obesity?"
    expected_json = {
        "patient_name": "True",
        "patient_doctor": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "Obesity",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 19 - Retrieve patient Jennifer Nguyen's blood type, doctor, and medical condition
def test_retrieve_jennifer_nguyen_info():
    query = "I need the blood type, doctor, and medical condition for patient Jennifer Nguyen."
    expected_json = {
        "patient_name": "Jennifer Nguyen",
        "patient_blood_type": "True",
        "patient_doctor": "True",
        "patient_medical_condition": "True",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json


# Test 20 - Retrieve patient information for a patient treated by Dr. Brian Chandler
def test_retrieve_patient_info_for_dr_brian_chandler():
    query = "What is the name, hospital, and medical condition for a patient treated by Dr. Brian Chandler?"
    expected_json = {
        "patient_name": "True",
        "patient_hospital": "True",
        "patient_medical_condition": "True",
        "patient_doctor": "Dr. Brian Chandler",
    }
    llm_response = format_query_to_json_with_llm_for_ner(query=query)
    prepared_answer = extract_xml_tags(
        xml_tags="prepared_answer", xml_string=llm_response
    )
    json_prepared_answer = string_to_dict(prepared_answer)
    assert json_prepared_answer == expected_json
