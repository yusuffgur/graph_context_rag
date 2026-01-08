import os
import click
import ast
import pandas as pd
import requests
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    BleuScore,
    RougeScore,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

BASE_URL = os.getenv("BASE_URL")
LOGIN_URL = os.getenv("LOGIN_URL")
TENANT_ID = os.getenv("TENANT_ID")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CLIENT_SCOPE = os.getenv("CLIENT_SCOPE")
REDIRECT_URI = os.getenv("REDIRECT_URI")
STORE_ID = os.getenv("CONTENT_STORE_ID")
# other configuration
azure_configs = {
    "base_url": "https://genaiapimna-dev.jnj.com/openai-chat",
    "emb_base_url": "https://genaiapimna-dev.jnj.com/openai-embeddings",
    "model_deployment": "gpt-4o",
    "model_name": "gpt-4o",
    "embedding_deployment": "text-embedding-ada-002",
    "embedding_name": "text-embedding-ada-002",
}
my_azure = AzureChatOpenAI(
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["base_url"],
    azure_deployment=azure_configs["model_deployment"],
    model=azure_configs["model_name"],
    validate_base_url=False,
)
my_azure_emb = AzureOpenAIEmbeddings(
    api_key=os.getenv("OPENAI_EMB_API_KEY"),
    openai_api_version="2023-05-15",
    azure_endpoint=azure_configs["emb_base_url"],
    azure_deployment=azure_configs["embedding_deployment"],
    model=azure_configs["embedding_name"],
)


def init_model(model_name):
    return AzureChatOpenAI(
        openai_api_version="2023-05-15",
        azure_endpoint=azure_configs["base_url"],
        azure_deployment=model_name,
        validate_base_url=False,
    )


def exchange_code_for_token(auth_code):
    # Constructing the token URL
    token_url = f"{LOGIN_URL}/{TENANT_ID}/oauth2/v2.0/token"

    # Constructing the headers
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    # Constructing the body
    body = {
        "client_id": CLIENT_ID,
        "scope": CLIENT_SCOPE,
        "code": auth_code,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
        "client_secret": CLIENT_SECRET,
    }

    # Making the POST request
    response = requests.post(token_url, headers=headers, data=body)

    # If the request was successful, the status code will be 200
    if response.status_code == 200:
        # The response is a JSON string, parse it to get the access token and refresh token
        response_json = json.loads(response.text)
        print(response_json)
        access_token = response_json["access_token"]

        # Store the tokens for future use
        with open(".credentials", "w") as file:
            json.dump(response_json, file)

        click.echo("Authentication successful. Tokens stored.")
    else:
        click.echo("Failed to authenticate.")


def get_content_stores(access_token):
    """Get a list of content stores."""
    url = f"{BASE_URL}/api/csphere/editor/contentstore/list"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["data"]


def get_knowledge_bases(access_token, store_id):
    """Get a list of knowledge bases for a specific content store."""
    url = (
        f"{BASE_URL}/api/csphere/editor/{store_id}/knowledge-base/list?skip=0&limit=20"
    )
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["data"]


def get_folders(access_token, store_id, kb_id):
    """Get a list of folders for a specific knowledge base."""
    url = f"{BASE_URL}/api/csphere/editor/{store_id}/folder/list?knowledge_base_id={kb_id}"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    data = response.json()["data"]
    return data if isinstance(data, list) else []


def create_knowledge_base(access_token, store_id, kb_name):
    """Create a new knowledge base."""
    url = f"{BASE_URL}/api/csphere/editor/{store_id}/knowledge-base/create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = json.dumps({"kbName": kb_name})
    response = requests.post(url, headers=headers, data=payload)
    return response.json()["data"][0]["KBId"]


def create_folder(access_token, store_id, kb_id, folder_name):
    """Create a new folder."""
    url = f"{BASE_URL}/api/csphere/editor/{store_id}/folder/create"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = json.dumps(
        {
            "FolderName": folder_name,
            "knowledgeBaseId": kb_id,
            "source": "genaics-cli tool",
        }
    )
    response = requests.post(url, headers=headers, data=payload)
    return response.json()["data"][0]["FolderId"]


def train_model(access_token, store_id, document_id, folder_id, kb_id):
    """Train the model after file upload."""
    url = f"{BASE_URL}/api/csphere/editor/{store_id}/train"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = json.dumps(
        {"documentId": document_id, "folderId": folder_id, "kbId": kb_id}
    )
    response = requests.post(url, headers=headers, data=payload)
    try:
        return response.json()
    except json.JSONDecodeError:
        print("Failed to parse the response from the API.", response.text)
        return None


def start_new_session(access_token, store_id):
    """Start a new session."""
    url = f"{BASE_URL}/api/csphere/user/{store_id}/session/new"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    return response.json()["data"]["sessionID"]


def get_suggested_questions(access_token, store_id, session_id):
    url = f"{BASE_URL}/api/csphere/user/{store_id}/suggestedQuestions?sessionId={session_id}"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    data = {"selectedRows": []}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    suggestions_str = response.json()["data"]["suggestions"]
    return ast.literal_eval(suggestions_str)


def get_answer(access_token, session_id, prompt):
    """Get an answer to a prompt."""
    url = f"{BASE_URL}/api/csphere/user/7087c084-5da0-4bb8-87b2-dd2021696651/{session_id}/questionAnswering"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = json.dumps({"prompt": prompt})
    response = requests.post(url, headers=headers, data=payload)
    return response.json()["data"]["data"]["answer"]


def get_full_response(access_token, store_id, session_id, prompt):
    """Get the full response to a prompt."""
    url = f"{BASE_URL}/api/csphere/user/{store_id}/{session_id}/questionAnswering"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_token}",
    }
    payload = json.dumps({"prompt": prompt})
    response = requests.post(url, headers=headers, data=payload)
    if response.status_code == 200:
        return response.json()["data"]["data"]
    else:
        raise Exception(f"Error: {response.text}")


def load_pdf_documents(file_paths):
    documents = []
    for file in file_paths:
        loader = PyPDFLoader(file_path=file)
        pages = loader.load_and_split()
        documents.extend(pages)
    return documents


def generate_dataset(documents, testset_size, model_to_test):
    my_model_to_test = init_model(model_to_test)
    generator_llm = LangchainLLMWrapper(my_model_to_test)

    generator_embeddings = LangchainEmbeddingsWrapper(my_azure_emb)

    generator = TestsetGenerator(generator_llm, generator_embeddings)
    dataset = generator.generate_with_langchain_docs(
        documents, testset_size=testset_size
    )
    test_df = dataset.to_pandas()

    return test_df


def prepate_dataset(file_path, test_size, model_name, access_token, session_id):
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        click.echo(f"Error: {e}")

    question_answering_data = {
        "retrieve_additional_contents": "False",
        "context": [],
        "collection": "cs-ehs-pal-test",
        "selected_content": ["/cs-ehs-pal-test"],
        "top_n": 3,
        "model_name": model_name,
    }

    data_samples = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    if test_size > len(df):
        raise Exception("Test size can't be bigger than your dataset.")
    # for data_piece in data['examples']:
    for i in range(test_size):
        data_samples["question"].append(df["user_input"][i])
        data_samples["ground_truth"].append(df["reference"][i])

        response = get_full_response(
            access_token, STORE_ID, session_id, df["user_input"][i]
        )

        data_samples["answer"].append(response["answer"])
        data_samples["contexts"].append([])
        for chunk in response["context"]:
            data_samples["contexts"][i].append(chunk["node_content"])

    dataset = Dataset.from_dict(data_samples)
    return dataset


def test_ragas(dataset, model_name):
    my_model = init_model(model_name)
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            BleuScore(),
            RougeScore(),
        ],
        llm=my_model,
        embeddings=my_azure_emb,
    )
    return result


def write_to_excel(results, file_name):
    writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    results.to_excel(writer, sheet_name="results", index=False)  # send df to writer
    workbook = writer.book
    text_wrap_format = workbook.add_format({"text_wrap": True})
    worksheet = writer.sheets["results"]  # pull worksheet object
    for idx, col in enumerate(results):  # loop through all columns
        worksheet.set_column(idx, idx, None, text_wrap_format)
    worksheet.autofit()
    writer.close()