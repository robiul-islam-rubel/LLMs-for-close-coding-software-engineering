from helper import *

# Function to send a prompt to Ollama with a specified model
def send_prompt_to_ollama(prompt, model):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,  
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(url, json=payload)
    output = response.json().get("response", "").strip()
    
    return output


# Different models for each agent
model_1 = "llama3:70b"    
model_2 = "llama3:70b-instruct"     
model_3 = "llama3:8b" 
question = "What do you think the relationship between AIBOMs and DataBOMs should be?"

def agent_1_decision(survey_text, code, definition, model, question=question):
   prompt = f"""
        You are an expert in analyzing and coding survey responses.
        Your task is to determine whether a given survey response perfectly matches a specific code definition.

        # Here's how to proceed:

        1. Carefully read and understand the survey question and the survey response.
        2. Then, review the provided code and its definition.
        3. Compare the survey response with the code definition.

        # Evaluate the match using the following strict criteria:

        - If the survey response clearly, directly, and unambiguously matches the code definition, return `"yes"`.
        - If the match is partial, vague, or requires inference or guessing, return `"no"`.
        - Only return `"yes"` for perfect and strong matches.
        - If you are uncertain or the response is open to interpretation, return `"no"`.

        # Here is the input:

        - Survey Question: {question}
        - Survey Response: {survey_text}
        - Code: {code}
        - Code Definition: {definition}

       # Respond with only one of the following JSON objects, with no extra text, explanation, or commentary:

        {{ "answer": "yes" }}  
        OR  
        {{ "answer": "no" }}
        """
   return send_prompt_to_ollama(prompt, model)

def agent_2_decision(survey_text, code, definition, model, question=question):
    prompt = f"""
            You are a survey coding analyst.
            Your task is to assess whether the survey response fully and explicitly matches the provided code definition.

            # Follow these steps:

            1. Read and understand the survey question to grasp the context.
            2. Analyze the survey response and the code definition carefully.
            3. Determine if the response directly supports and aligns with the code definition.

            # Evaluation Criteria:

            - A match must be clear, complete, and unambiguous.
            - There should be no need for assumptions, inference, or interpretation.
            - If the response perfectly and explicitly reflects the code definition, respond with "yes".
            - If the response is vague, partial, or only loosely related, respond with "no".
            - When in doubt, always choose "no".

            # Input:

            - Survey Question: {question}
            - Survey Response: {survey_text}
            - Code: {code}
            - Code Definition: {definition}

       Respond with only one of the following JSON objects, with no extra text, explanation, or commentary:

            {{ "answer": "yes" or "no" }}
         """

    return send_prompt_to_ollama(prompt, model)

def agent_3_decision(survey_text, code, definition, model, question=question):
    prompt = f"""
        You are a highly precise survey coding expert.
        Your task is to decide whether the following survey response clearly and directly matches the provided code definition.

        #Please follow these steps:
        1. Review the survey question to understand the context.
        2. Carefully read the survey response.
        3. Analyze the code and its definition.
        4. Decide if the response aligns fully with the code definition.

        # Guidelines for Evaluation:
        - Respond with "yes" only if the survey response clearly, directly, and unambiguously matches the code definition.
        - Do not guess or assume. The match must be exact and obvious.
        - If the response is partial, vague, indirect, or requires interpretation, respond with "no".
        - If there is any uncertainty, respond with "no".
        - Accept only perfect and strong matches.

        # Input:

        - Survey Question: {question}
        - Survey Response: {survey_text}
        - Code: {code}
        - Code Definition: {definition}

        Return your answer strictly in the following format:

        {{ "answer": "yes" or "no" }}
        """

    return send_prompt_to_ollama(prompt, model)


def process_survey_responses(survey_csv_file, code_csv_file, output_file):
    survey_df = pd.read_csv(survey_csv_file, encoding="ISO-8859-1")
    
    if "survey_response" not in survey_df.columns:
        raise ValueError(f"Missing 'survey_response' column in {survey_csv_file}")

    code_dict = {}
    code_df = pd.read_csv(code_csv_file, encoding="ISO-8859-1")
    
    if "code" not in code_df.columns or "definition" not in code_df.columns:
        raise ValueError(f"Missing 'code' or 'definition' column in {code_csv_file}")
    
    for code, desc in zip(code_df["code"], code_df["definition"]):
        code_dict[code] = desc
    response_code_mapping = []
    for survey_text in survey_df["survey_response"]:
        ans = None
        matched_code = []
        no = 0
        itr = 0
        for code, definition in code_dict.items():
            itr = itr + 1
            yes_vote = 0
            agent1_1 = agent_1_decision(survey_text, code, definition, model_1)
            agent1_2 = agent_1_decision(survey_text, code, definition, model_2)
            agent1_3 = agent_1_decision(survey_text, code, definition, model_3)
            print(f"{model_3}: {agent1_3}")

            agent2_1 = agent_2_decision(survey_text, code, definition, model_1)
            agent2_2 = agent_2_decision(survey_text, code, definition, model_2)
            agent2_3 = agent_2_decision(survey_text, code, definition, model_3)

            agent3_1 = agent_3_decision(survey_text, code, definition, model_1)
            agent3_2 = agent_3_decision(survey_text, code, definition, model_2)
            agent3_3 = agent_3_decision(survey_text, code, definition, model_3)

            match1_1 = check_match(agent1_1)
            match1_2 = check_match(agent1_2)
            match1_3 = check_match(agent1_3)

            match2_1 = check_match(agent2_1)
            match2_2 = check_match(agent2_2)
            match2_3 = check_match(agent2_3)

            match3_1 = check_match(agent3_1)
            match3_2 = check_match(agent3_2)
            match3_3 = check_match(agent3_3)

            if match1_1== "yes":
                yes_vote = yes_vote+1
            if match1_2== "yes":
                yes_vote = yes_vote+1
            if match1_3 == "yes":
                yes_vote = yes_vote+1
            
            if match2_1== "yes":
                yes_vote = yes_vote+1
            if match2_2== "yes":
                yes_vote = yes_vote+1
            if match2_3 == "yes":
                yes_vote = yes_vote+1
            
            if match3_1== "yes":
                yes_vote = yes_vote+1
            if match3_2== "yes":
                yes_vote = yes_vote+1
            if match3_3 == "yes":
                yes_vote = yes_vote+1
            print(f"Total Yes Vote: {yes_vote}")
            if yes_vote>=6:
                matched_code.append(code)
                
            print(f"Survey Text: {survey_text}, Code: {code}")
            print(f"Match1_1: {match1_1}, Match1_2: {match1_2}, Match1_3: {match1_3}")
            print(f"Match2_1: {match2_1}, Match2_2: {match2_2}, Match2_3: {match2_3}")           
            print(f"Match3_1: {match3_1}, Match3_2: {match3_2}, Match3_3: {match3_3}")           

            # matched_code.append(code_from_codebook)           
            print("Iteration: \n",itr)
        print(f"Python list: {matched_code}")
        response_code_mapping.append([survey_text,matched_code])
   
    df_output = pd.DataFrame(response_code_mapping, columns=["survey_response","llama_generated"])
    df_output["human_codes"] = survey_df["human_codes"]
    df_output.to_csv(output_file, index=False)
    
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    # # Example Usage:
    survey_text = "Lack of parts supply"
    code = "Shortages"
    definition = "The inability to acquire components due to shortages"

    # Get the final decision
    # agent_1_decision = agent_1_decision(survey_text, code, definition,model_1)
    # agent_2_decision = agent_2_decision(survey_text, code, definition,model_2)
    # agent_3_decision = agent_3_decision(survey_text, code, definition,model_3)

    # print(f"Agent1: {agent_1_decision}, Agent2: {agent_2_decision}, Agent3: {agent_3_decision}")

    # process_survey_responses(
    #     "dataset/cyber_physical/o2/survey.csv",
    #     "dataset/cyber_physical/o2/code_with_definition.csv",
    #    "dataset/cyber_physical/o2/survey.csv"
    # )

    # process_survey_responses(
    #     "dataset/sbom_ca/survey.csv",
    #     "dataset/sbom_ca/code_with_definition.csv",
    #    "dataset/sbom_ca/survey.csv"
    # )

    # process_survey_responses(
    #     "dataset/cyber_physical/g1/survey.csv",
    #     "dataset/cyber_physical/g1/code_with_definition.csv",
    #    "dataset/cyber_physical/g1/survey.csv"
    # )

    # process_survey_responses(
    #     "dataset/ml/d5/survey.csv",
    #     "dataset/ml/d5/code_with_definition.csv",
    #    "dataset/ml/d5/survey.csv"
    # )

    # Print the full conversation history
    # clean_json = json.dumps(multi_agent_decision(survey_text, code, definition, model_1, model_2, model_3),indent=2, ensure_ascii=False).replace("\n", " ").replace('\\n',"").replace("\\","")
    # print(json.dumps(clean_json))