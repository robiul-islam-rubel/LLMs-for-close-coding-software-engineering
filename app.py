
from test import process_survey_responses
from helper import *

st.title("Survey Coding Tool (LLaMA Multi-Agent Version)")

st.sidebar.header("Upload Files")

# Upload survey and code definition files
survey_file = st.sidebar.file_uploader("Upload Survey Responses CSV", type="csv")
code_file = st.sidebar.file_uploader("Upload Code Definitions CSV", type="csv")

# Input survey question (not actually used yet)
survey_question = st.text_input("Enter Survey Question Context")

if st.button("Start Coding"):
    if survey_file is not None and code_file is not None and survey_question:
        with st.spinner("Processing survey responses, this may take a while..."):
            
            # Save uploaded files temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_survey:
                temp_survey.write(survey_file.read())
                temp_survey_path = temp_survey.name
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_code:
                temp_code.write(code_file.read())
                temp_code_path = temp_code.name
            
            output_temp_path = temp_survey_path.replace(".csv", "_coded.csv")
            print("Check: ", temp_survey_path)
            df = pd.read_csv(temp_survey_path)
            print("Length: ", len(df) )
            # Call your backend function
            process_survey_responses(temp_survey_path, temp_code_path, output_temp_path,survey_question)
            
            # Load and show results
            result_df = pd.read_csv(output_temp_path)
            st.success("Processing Completed!")
            st.subheader("Coded Results")
            st.dataframe(result_df)

            # Download link
            st.download_button(
                label="Download Coded Results",
                data=result_df.to_csv(index=False),
                file_name="coded_results.csv",
                mime="text/csv"
            )
    else:
        st.error("Please upload both CSV files and enter the survey question.")
