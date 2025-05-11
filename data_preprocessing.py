from helper import *

def load_dataset():
    # Load json file
    data_path = "dataset/cyber_physical/cyber_physical.json"
    df = load_json_file(data_path)
    print(json.dumps(df,indent=4))
    return df

def extract_short_answer(df, key_target):
    all_fields = find_fields(df, key_target)
    print(f"Fields: {all_fields}, Length: {len(all_fields)}")
    survey_file_path = "dataset/cyber_physical/survey.csv"
    survey_data = pd.DataFrame({"survey_response": all_fields})
    survey_data.to_csv(survey_file_path,index=False)



if __name__=="__main__":

    # fields = ["AI7","D5","D7","D8","D9","C1","C2","C7"]
    # df = load_dataset()
    # extract_short_answer(df, "S4")
    # for field in fields:
    #     extract_short_answer(df, field)
    
    ## dataset cleaning
    # data_path = "dataset/sboms/survey_data_ml.csv"
    # df_clean = cleaning_data(data_path=data_path)
    # df_clean.to_csv(data_path,index=False)
    # print(df_clean)


    
    


    # Extract all AI7 fields
    # ai7_values = find_ai7_fields(df)
    # boms_data = "dataset/bomps.csv"

    # boms_data = pd.DataFrame({"survey_response": ai7_values})

    # Print results
    # for i, val in enumerate(ai7_values, 1):
    #     print(f"AI7 Value {i}: {val}")
    #     boms_data.loc[i,"survey_response"] = val
    # boms_data.to_csv("dataset/sboms/bomps.csv",index=False)
    
    # Load human coding
    # coding = pd.read_csv("dataset/response_coding.csv")
    # print(coding.head())

    source_file = "dataset/cyber_physical/response_coding.csv"
    target_file = "dataset/cyber_physical/o2/survey.csv"
    target_column = "O2"
    output_file = target_file
    new_column_name = "human_codes"
    ## Copy one column to another csv file
    # copy_column(source_file,target_file,target_column,new_column_name,output_file)
    

    # target_column = "AI7"
    # bomps["human_codes"] = coding[target_column]
    # bomps.to_csv(bomps,index=False)

   # Extracting code and its definition

    data_path = "dataset/codes.json"   
    df = load_json_file(data_path)
    outer_key = "CPS"
    inner_key = "O2 (Q5)"
    values = extract_value(df,outer_key,inner_key)
    # print(ai7_values)
    data_path = "dataset/cyber_physical/o2/code_with_definition.csv"
    data = pd.read_csv(data_path)
    new_data = pd.DataFrame(list(values.items()), columns=["code", "definition"])
    # for i,(key, value) in enumerate(values.items()):
    #     print(f"Key: {key}")
    #     data.loc[i,"code"] = key 
    #     data.loc[i,"definition"] = value
    data = pd.concat([data,new_data],ignore_index=True)
    # data.to_csv(data_path,index=False)

    
    # Remove column
    df =  pd.read_csv("dataset/cyber_physical/g1/survey.csv")
    df =  df.drop(["llama_generated"],axis=1)
    # df.to_csv("dataset/cyber_physical/g1/survey.csv",index=False)

    # df = pd.read_csv("dataset/sboms/bomps.csv")
    # for i in range(len(df)):
    #     data = df['survey_response'][i]
    #     print(f"Before cleaning: {data}")
    #     data = data.replace("\n","").replace("-","")
    #     print(f"After cleaning: {data}")

    # Removing unnecessary stuffs such as \n, - or etc.
    # df = cleaning_data()
    # print(df)
    # df.to_csv("dataset/sboms/bomps.csv",index=False)
    # print(df)


    # Extract the code and definitions
    # data_path = "dataset/sboms/codes.json"
    # outer_key = "ML"
    # fields = ["AI7", "D5", "D7", "D8", "D9", "C1", "C2", "C7"]
    # output_csv = "dataset/sboms/boms_code_with_definition.csv"

    # extract_and_save_values(data_path, outer_key, fields, output_csv)


    # df = pd.read_csv("dataset/sboms/boms_code_with_definition.csv")
    # print(df)

    # HCI data pre-processing
    # df = pd.read_csv("dataset/hci/survey.csv", encoding="ISO-8859-1")
    # df = df[["why_art","human_codes"]]
    # df.to_csv("dataset/hci/survey.csv",index=False)
   


