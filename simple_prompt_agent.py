
import argparse

from langchain.chat_models import ChatOpenAI

from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
import os
from dotenv import load_dotenv

from langchain.callbacks import get_openai_callback
load_dotenv()


class SinglePromptAgent:
    def __init__(self,
                temperature: int = 0.5, 
                openai_api_key: str = '',
                template: str = 
                '''
                Your are an useful assistant.

                User message: {user_message}
                ''',
                model: str= 'gpt-4'):
        self.system_template = template
        # self.human_template = '{letter_to_format}'
        # The temperature parameter controls the randomness of the model's output, with a lower temperature resulting in more deterministic output.
        self.chat = ChatOpenAI(temperature=temperature, model=model)



        # You can use the SystemMessagePromptTemplate.from_template to transform a single text
        # into a prompt with variables
        self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.system_template)

        # self.user_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)


        # Combine the system and human prompts into a single chat prompt.
        # Availiable schemas are:
        # SystemMessagePromptTemplate,
        # HumanMessagePromptTemplate,
        # AIMessagePromptTemplate

        self.chat_prompt = ChatPromptTemplate.from_messages(
            # [self.system_message_prompt, self.user_message_prompt]
            [self.system_message_prompt]
        )

        
        # Create an instance of LLMChain
        self.chain = LLMChain(llm=self.chat, prompt=self.chat_prompt)
    

    def run(self,**args):
        agent_response = self.chain.run(
            **args
        )

        return agent_response