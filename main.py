import os
import argparse

import pinecone  
from dotenv import load_dotenv

from src.utils import get_directory_content, embed_text
from text_loader import CustomTextLoader
from simple_prompt_agent import SinglePromptAgent
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

# First we need to load our environment variables
load_dotenv()

def split_text_in_smaller_chunks(list_of_files_to_save, chunk_size=2000, chunk_overlap=100):
    list_of_documents_chunks = []
    
    for f in list_of_files_to_save:
        loader = CustomTextLoader(plain_text=f["file_content"], metadata={
            "file_name": f["file_name"]
        })
        list_of_documents_chunks.extend(loader.load())
        
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
            )
    splits = text_splitter.split_documents(list_of_documents_chunks)
    return splits

def generate_vector_tuples(chunks):
    vector_tuples = []
    for index, c in enumerate(chunks):
        vector = embed_text(c.page_content)
        unique_vector_id = f'{index}-{c.metadata["file_name"]}'
        c.metadata["text"] = c.page_content
        vector_tuple = (unique_vector_id, vector, c.metadata)
        vector_tuples.append(vector_tuple)
    
    return vector_tuples


def main(index_name:str='db-docs',load_files=False):
    # Init the pinecone-client with our credentials
    pinecone.init(      
        api_key=os.environ.get("PINECONE_API_KEY"),      
        environment=os.environ.get("PINECONE_ENVIRONMENT"),       
    )      

    # We need to use an index to perform operations, so we can create a new index or use one our indexes
    
    # If the index don't exist, then we create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=1536,
            metric='cosine'
        )

    # This instance of Index has built-in methods for us
    index = pinecone.Index(index_name)

    if load_files:
        # Get the files, objects, implement your own logic to get the content you need
        list_of_files_to_save = get_directory_content('docs')
        print(f"Obtained {len(list_of_files_to_save)} files in the directory")

        # Split the text from each file in smaller chunks, this is important to pass less context to the LLM later
        chunks_with_metadata = split_text_in_smaller_chunks(list_of_files_to_save=list_of_files_to_save)
        print(f"Obtained {len(chunks_with_metadata)} chunks from the files")

        vector_tuples_to_upsert = generate_vector_tuples(chunks=chunks_with_metadata)

        # The upsert method of the index dictionary can be used to add or update new vector with certain metadata
        index.upsert(vectors=vector_tuples_to_upsert)

    while True:
        # Now we need to test our index, so, lets vectorize a question
        question = input("\nLet's ask something about your data!: ")

        if 'break()' in question:
            break
        vector_question = embed_text(question)

        # We can use the query method of the index instance to get the most relevant vector to 
        # our vector_question
        result_from_pinecone = index.query(
            vector=vector_question, top_k=2,include_metadata=True
        )

        context = ""
        for matched_vector in result_from_pinecone.matches:
            # The response has a matches attribute with the top_k similar vectors

            # Get the context stored in the metadata text key
            text_from_vector = matched_vector.get("metadata").get("text")

            # Add the context to a variable to pass it to the LLM (you can change this logic for a different one)
            context += text_from_vector + "\n\n"

        # Add more variables, change this prompt to satisfy your requirements
        template = """You are an useful assistant.

        Answer this question carefully:
        {question}

        This context can be helpful for you.:
        {context}
        """

        manager = SinglePromptAgent(template=template)

        print(manager.run(context=context, question=question))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_files", help="Load files flag", action="store_true")
    parser.add_argument("--index_name", help="Index to perform operations")
    args = parser.parse_args()
    if args.index_name:
        main(index_name=args.index_name, load_files=args.load_files)
    else:
        main(load_files=args.load_files)
