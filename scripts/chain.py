import streamlit as st
import os
import pandas as pd
import logging
from dotenv import load_dotenv

# Charger les secrets depuis Streamlit
def main():
    # Récupérer la clé API depuis les secrets de Streamlit
    openai_api_key = st.secrets["OPEN_API_KEY"]

    if not openai_api_key:
        st.error("OPEN_API_KEY is not set. Please set it in the Streamlit secrets.")
    else:
        st.success("API Key loaded successfully.")

    # Vérifiez et importez matplotlib.pyplot
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        st.error("Matplotlib is not installed. Please install it by running `pip install matplotlib`.")
        raise e

    from sklearn.cluster import KMeans
    from tslearn.utils import to_time_series_dataset
    from tslearn.clustering import TimeSeriesKMeans
    from prophet import Prophet
    from langchain_core.tools import tool
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_experimental.agents import create_csv_agent
    from langchain.agents.agent_types import AgentType
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda, RunnablePassthrough
    from operator import itemgetter
    import json
    from pydantic import ValidationError

    # Configuration du logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Définition des outils
    @tool
    def load_time_series(file_path: str) -> pd.DataFrame:
        """Load time series data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
            df.set_index('date', inplace=True)
            return df
        except Exception as e:
            logging.error("Error loading time series data from file %s: %s", file_path, e)
            raise

    @tool
    def calculate_monthly_average_temperature(file_path: str) -> pd.DataFrame:
        """Calculate monthly average temperature from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
            df.set_index('date', inplace=True)
            monthly_avg = df.resample('ME').agg({'tre200s0': 'mean'})
            return monthly_avg.reset_index()
        except Exception as e:
            logging.error("Error processing file %s: %s", file_path, e)
            raise

    @tool
    def kmeans_cluster_time_series(file_path: str, n_clusters: int) -> pd.DataFrame:
        """Apply KMeans clustering to time series data from a CSV file."""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
            df.set_index('date', inplace=True)
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(df[['tre200s0']].values.reshape(-1, 1))
            df['cluster'] = clusters
            return df.reset_index()
        except Exception as e:
            logging.error("Error during KMeans clustering: %s", e)
            raise

    @tool
    def cluster_temperatures_tslearn(file_path: str, n_clusters: int = 4) -> pd.DataFrame:
        """Cluster the temperatures using time series clustering with tslearn."""
        try:
            data = pd.read_csv(file_path)
            data['date'] = pd.to_datetime(data['year'] * 1000 + data['day_of_year'], format='%Y%j') + pd.to_timedelta(data['minute_of_day'], unit='m')
            data.set_index('date', inplace=True)
            data['month'] = data.index.month
            monthly_avg_temp = data.groupby(['name', 'month'])['tre200s0'].mean().reset_index()
            pivot_monthly_avg_temp = monthly_avg_temp.pivot(index='name', columns='month', values='tre200s0')
            pivot_monthly_avg_temp_filled = pivot_monthly_avg_temp.fillna(pivot_monthly_avg_temp.mean())
            formatted_dataset = to_time_series_dataset(pivot_monthly_avg_temp_filled.to_numpy())
            model = TimeSeriesKMeans(n_clusters=n_clusters, metric="euclidean", random_state=33)
            labels = model.fit_predict(formatted_dataset)
            result_df = pd.DataFrame({'name': pivot_monthly_avg_temp_filled.index, 'cluster': labels})
            
            plt.figure(figsize=(10, 6))
            for i, center in enumerate(model.cluster_centers_):
                plt.plot(center.ravel(), label=f'Cluster {i}')
            plt.title('Centres des Clusters de Température Moyenne Mensuelle par Station')
            plt.xlabel('Mois')
            plt.ylabel('Température Moyenne (°C)')
            plt.xticks(ticks=range(12), labels=range(1, 13))
            plt.legend()
            plt.show()
            
            return result_df
        
        except Exception as e:
            logging.error("Error clustering temperatures with tslearn: %s", e)
            raise

    @tool
    def predict_future_temperatures(file_path: str, periods: int = 12) -> pd.DataFrame:
        """Predict future temperatures using the Prophet model."""
        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['year'] * 1000 + df['day_of_year'], format='%Y%j') + pd.to_timedelta(df['minute_of_day'], unit='m')
            df.set_index('date', inplace=True)
            df = df.resample('ME').agg({'tre200s0': 'mean'}).reset_index()
            df.rename(columns={'date': 'ds', 'tre200s0': 'y'}, inplace=True)
            
            model = Prophet()
            model.fit(df)
            
            future = model.make_future_dataframe(periods=periods, freq='M')
            forecast = model.predict(future)
            
            fig1 = model.plot(forecast)
            plt.title('Forecasted Temperatures')
            plt.xlabel('Date')
            plt.ylabel('Temperature')
            plt.show()
            
            return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        except Exception as e:
            logging.error("Error predicting future temperatures: %s", e)
            raise

    # Initialiser les modèles OpenAI avec la clé API
    llm_general = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key)
    llm_tools = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)

    # Définir les outils
    tools = [load_time_series, calculate_monthly_average_temperature, kmeans_cluster_time_series, cluster_temperatures_tslearn, predict_future_temperatures]

    # Générer les descriptions des outils
    rendered_tools = "\n".join([f"{tool.name}: {tool.__doc__}" for tool in tools])

    # Définir le prompt du système pour les outils
    system_prompt_tools = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:

    {rendered_tools}

    Given the user input, decide whether to answer the question directly or to use one of the tools. If you decide to use a tool, return the name and input of the tool to use as a JSON blob with 'name' and 'arguments' keys. If you decide to answer directly, return the answer as a string."""

    # Créer le template de prompt pour les outils
    prompt_tools = ChatPromptTemplate.from_messages(
        [("system", system_prompt_tools), ("user", "{input}")]
    )

    # Créer la chaîne pour les outils
    chain_tools = prompt_tools | llm_tools | JsonOutputParser()

    # Définir la fonction pour appeler les outils
    def tool_chain(model_output):
        if isinstance(model_output, str):
            return model_output  # Return the direct answer
        tool_map = {tool.name: tool for tool in tools}
        chosen_tool = tool_map[model_output["name"]]
        tool_args = model_output["arguments"]
        tool_args['file_path'] = file_path  # Ajout du chemin vers le fichier CSV
        return chosen_tool(tool_args)

    chain_tools = prompt_tools | llm_tools | JsonOutputParser() | RunnableLambda(tool_chain)

    # Créer l'agent CSV pour les questions générales
    def create_csv_agent_for_file(file_path):
        return create_csv_agent(
            ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key),
            file_path,
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    # Modèle pour choisir l'agent approprié
    selection_prompt = """You are an assistant that decides whether to use a tool or the CSV agent. Given the user input, decide whether the question requires using one of the tools or can be answered directly using the CSV agent. Respond with "use the tool" if a tool is needed, otherwise respond with "use the CSV agent"."""
    selection_chain = ChatPromptTemplate.from_messages(
        [("system", selection_prompt), ("user", "{input}")]
    ) | llm_tools

    # Fonction pour choisir l'agent approprié
    def choose_agent(question, file_path):
        response = selection_chain.invoke({"input": question})
        if "use the tool" in response.content.lower():
            return chain_tools
        else:
            return create_csv_agent_for_file(file_path)

    # Combiner les chaînes
    def combined_chain(question, file_path):
        agent = choose_agent(question, file_path)
        return agent.invoke({"input": question})

    # Fonction pour afficher les résultats
    def print_results(results):
        if isinstance(results, str):
            st.write(results)
        else:
            st.dataframe(results)

    # Application Streamlit
    st.title('Weather Data Analysis with LangChain')

    # Télécharger un fichier CSV
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        # Créer le dossier uploads s'il n'existe pas
        if not os.path.exists("uploads"):
            os.makedirs("uploads")

        # Enregistrer le fichier téléchargé
        file_path = os.path.join("uploads", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded {uploaded_file.name}")

        # Boîte de question unique
        question = st.text_input("Enter your question about the data")
        if st.button("Ask Question"):
            try:
                response = combined_chain(question, file_path)
                print_results(response)
            except ValidationError as e:
                st.error(f"Validation error: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning("Please upload a CSV file to proceed.")

if __name__ == "__main__":
    main()
