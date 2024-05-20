import streamlit as st
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
from dotenv import load_dotenv
import os

# Ajout des images et du texte explicatif en haut de la page
st.markdown('''
![alt text](https://moodle.msengineering.ch/pluginfile.php/1/core_admin/logo/0x150/1643104191/logo-mse.png "MSE Logo") 
![alt text](https://www.hes-so.ch/typo3conf/ext/wng_site/Resources/Public/HES-SO/img/logo_hesso_master_tablet.svg "Hes Logo")

Cette application a été développée dans le cadre d'un projet d'approfondissement visant à traiter les séries temporelles avec des modèles de langage (LLM). Elle utilise un agent CSV de la bibliothèque LangChain. Cette application permet aux utilisateurs d'interagir avec des fichiers CSV. Par exemple, elle peut répondre à des questions telles que "Combien y a-t-il de stations ?" ou "Quelle est la ville la plus chaude ?".
''')

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
