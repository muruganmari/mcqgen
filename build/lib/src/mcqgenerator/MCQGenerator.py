import os
import json
import pandas as pa
import traceback
from  langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()
KEY ="AIzaSyBtvynQSdtzsD3ris0sIlCfHGqNji-gY0k"
llm = GoogleGenerativeAI(model = "gemini-2.0-flash", api_key=KEY)
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.callbacks import get_openai_callback
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.tracers import ConsoleCallbackHandler  # For debugging
from langchain.chains import SequentialChain

callback_manager = CallbackManager([ConsoleCallbackHandler()])

RESPONSE_JSON = {
    "1": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "2": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
    "3": {
        "mcq": "multiple choice question",
        "options": {
            "a": "choice here",
            "b": "choice here",
            "c": "choice here",
            "d": "choice here",
        },
        "correct": "correct answer",
    },
}

TEMPLATE="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_gereration_prompt =PromptTemplate(
    input_variables = ['text', 'number', 'subject', 'tone', 'response_json'],
    template = TEMPLATE
)

quiz_chain = LLMChain(llm=llm,prompt=quiz_gereration_prompt,output_key='quiz', callback_manager=callback_manager,verbose=True )

TEMPLATE2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""

quiz_evaluation_prompt = PromptTemplate(input_variables=['subject', 'quiz'], template=TEMPLATE2)

review_chain = LLMChain(llm=llm, prompt = quiz_evaluation_prompt, output_key='review',callback_manager=callback_manager, verbose=True)

generate_evaluate_chain = SequentialChain(chains=[quiz_chain, review_chain], input_variables=['text', 'number', 'subject', 'tone', 'response_json'],
                                          output_variables=['quiz', 'review'], callback_manager=callback_manager,verbose=True)