# Healthcare Conversational Agent

A chat assistant that lets hospital doctors look up patient information in plain language, and that refuses toxic or off-topic requests.

I built it with LangChain and Claude 3 Haiku on AWS Bedrock, to learn how to make an LLM agent reliable enough to put in front of users. The data is synthetic.

## How a question is handled

Take the question "What is the blood type and medical condition of Samantha Rodriguez?".

1. **The agent picks a tool.** It follows the [ReAct](https://react-lm.github.io/) pattern: it reasons about the question, then chooses one of four tools. Here it chooses the data search.
2. **A second LLM call turns the question into a query.** It extracts the entities and marks what is being asked for:
   ```python
   {'patient_name': 'Samantha Rodriguez', 'patient_blood_type': 'True', 'patient_medical_condition': 'True'}
   ```
3. **Plain Python does the lookup.** The dictionary is used to filter the dataset with pandas. The model never writes or runs a query itself.
4. **The agent writes the reply** from the rows that came back.

The other three tools cover what the agent should not answer, or cannot answer alone:

| The doctor says | Tool that fires | What happens |
|---|---|---|
| "I hate all patients with mental health issues." | Toxicity Police | The agent refuses and steers back to the use case |
| "How's the weather today?" | Irrelevancy | The agent says this is outside its scope |
| "Who was the first patient with blood type B+?" (after earlier questions) | Conversation history | The agent answers from earlier turns |

## Design choices

- **The model never touches the data directly.** It only produces a small dictionary over five allowed fields (name, blood type, doctor, hospital, medical condition). Anything outside those fields cannot be queried, whatever the prompt says.
- **Guardrails are tools, and toxicity has priority.** Requests for forbidden patient data are treated as toxic, alongside hate speech and insults.
- **Broad requests are refused.** If a question would return the whole dataset, the agent asks the doctor to be more specific. Results are capped at five rows.
- **Prompts are tested like code.** 48 pytest cases check that the right tool fires and that the extraction returns the expected dictionary. A prompt edit or a model update that changes behaviour shows up as a failing test.
- **Prompts are written for Claude 3 Haiku**, using XML tags and a thinking step before the answer.

## Limits

- Lookups are exact matches, so a misspelled patient name finds nothing.
- Conversation memory is held in process memory with a single session, which is fine for a demo and not for several users.
- The tests call the real model, so they need AWS access, cost a little, and can occasionally vary between runs.
- It runs on LangChain 0.1 and Claude 3 Haiku. Dependencies are pinned to those versions and the project is not maintained.

## Run it

You need Python 3.10+, [Poetry](https://python-poetry.org/), and an AWS account with access to Claude 3 Haiku on Bedrock (region `us-east-1`).

```bash
poetry install
cp .env_template .env   # then set AWS_CREDENTIALS_PROFILE_NAME to an AWS CLI profile with Bedrock access
python -m streamlit run demo/streamlit_demo.py
```

Run the tests with `pytest`.

## Where things are

- `app/src/`: the agent and its tools
- `app/prompts/`: the agent prompt and the extraction prompt
- `app/utils/`: the pandas lookup and parsing helpers
- `tests/`: prompt tests for toxicity, irrelevancy and extraction
- `demo/`: the Streamlit chat interface
- `data/`: synthetic patient data from Kaggle's [Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)

## License

[MIT](./LICENSE)
