🔍 LLM for Close Coding
LLM for Close Coding is a research tool designed to automate the qualitative analysis process by assigning predefined codes to open-ended survey responses using Large Language Models (LLMs). It reduces the time and effort required for manual coding while improving consistency and scalability.

📌 Project Description
In qualitative research, close coding involves tagging textual responses with predefined categories based on a codebook. This project uses state-of-the-art LLMs (e.g., GPT, LLaMA, Gemini) to replicate this process automatically. Our system also supports multi-agent validation, model agreement analysis, and error checking.

Key features:

Upload survey responses and codebooks.

Apply LLMs to assign codes.

Perform validation using multiple models.

Export results for further analysis.

🧠 Core Technologies
Python 3.x

OpenAI GPT-4 (or alternatives like LLaMA, Gemini via API)

Pandas & NumPy

Streamlit (UI)

Ollama (optional, for local model execution)

🗃️ Project Structure
perl
Copy
Edit
llm-close-coding/
├── main.py                # Streamlit app interface
├── multi_agent.py         # Multi-model processing logic
├── helper.py              # Utility functions
├── prompts/               # Prompt templates for LLMs
├── data/
│   ├── sample_survey.csv
│   └── sample_codebook.csv
└── results/               # Output with coded responses
🚀 Getting Started
Prerequisites
Python 3.8+

OpenAI API key (or configured local LLM)

pip install -r requirements.txt

Running the App
bash
Copy
Edit
streamlit run main.py
Upload Format
Survey CSV: Contains free-text responses in a column.

Codebook CSV: Each row should define one code and its description.

✅ Example Use Case
Imagine you have 200 responses to the question:
“What challenges do you face when using AI tools in software development?”

This tool can automatically assign responses into categories like:

Trust issues

Debugging difficulty

Poor documentation

Over-reliance on AI

Saving you days of manual coding work.

🧪 Evaluation
Supports majority-vote validation across multiple models.

You can analyze agreement rates and extract mismatches.

Integrates human-in-the-loop error correction for refining labels.

🧩 Future Work
Add support for zero-shot vs. few-shot comparison.

Fine-tune smaller LLMs using coded datasets.

Support hierarchical codebooks.

📄 License
This project is for academic/research use. For commercial use, please contact the author.

🙌 Acknowledgements
Based on research at the intersection of LLMs and qualitative analysis in software engineering.

Inspired by tools like OpenAI GPT-4, CodeBERT, and related deductive coding frameworks.
