import os

from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


def get_directory_content(directory: str) -> list:
        """
        This function takes a directory name as input and returns a list of dictionaries.
        Each dictionary contains two keys: 'file_name' and 'file_content'.
        The 'file_name' key stores the name of each file in the directory,
        and the 'file_content' key stores the content of each file.
        """
        directory_content = []
        # Check if the directory exists
        if os.path.exists(directory):
                # Iterate over each file in the directory
                for file_name in os.listdir(directory):
                        file_path = os.path.join(directory, file_name)
                        # Check if the current item is a file
                        if os.path.isfile(file_path):
                                # Read the content of the file
                                with open(file_path, 'r') as file:
                                        file_content = file.read()
                                # Create a dictionary with file name and content
                                file_dict = {
                                        'file_name': file_name,
                                        'file_content': file_content
                                }
                                # Append the dictionary to the directory_content list
                                directory_content.append(file_dict)
        return directory_content





def embed_text(text:str):
    
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )
    texts = [text]

    res = embed.embed_documents(texts)
    return res[0]

def embed_list_of_text(texts:list[str]):
    
    model_name = 'text-embedding-ada-002'

    embed = OpenAIEmbeddings(
        model=model_name,
        openai_api_key=OPENAI_API_KEY
    )

    res = embed.embed_documents(texts)
    return res
