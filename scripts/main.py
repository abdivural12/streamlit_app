from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI  # Suppression de ChatOpenAI car il semble redondant
from dotenv import load_dotenv
import os
import streamlit as st

def main():
    load_dotenv()

    # Récupérer la valeur de la clé API depuis les secrets Streamlit
    api_key = st.secrets["OPEN_API_KEY"]

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        try:
            agent = create_csv_agent(
                OpenAI(api_key=api_key, temperature=0), csv_file, verbose=True)

            user_question = st.text_input("Ask a question about your CSV: ")

            if user_question:
                with st.spinner(text="In progress..."):
                    st.write(agent.run(user_question))
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
