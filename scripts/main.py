import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.llms import OpenAI  # Mise à jour de l'importation
from langchain_experimental.agents.agent_toolkits import create_csv_agent

def main():
    load_dotenv()  # Cela chargera les variables d'un fichier .env localement, mais sera ignoré en production

    # Récupérer la valeur de la variable d'environnement
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key is None:
        st.error("OPENAI_API_KEY is not set. Please set it in the .env file or in the environment variables.")
        return

    st.set_page_config(page_title="Ask your CSV")
    st.header("Ask your CSV 📈")

    csv_file = st.file_uploader("Upload a CSV file", type="csv")
    if csv_file is not None:
        try:
            agent = create_csv_agent(OpenAI(openai_api_key=api_key, temperature=0), csv_file, verbose=True)
        except Exception as e:
            st.error(f"Error initializing agent: {e}")
            return

        user_question = st.text_input("Ask a question about your CSV: ")

        if user_question:
            with st.spinner(text="In progress..."):
                try:
                    st.write(agent.run(user_question))
                except Exception as e:
                    st.error(f"Error running agent: {e}")

if __name__ == "__main__":
    main()
