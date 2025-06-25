# main.py

import os
from dotenv import load_dotenv
from kaggle_secrets import UserSecretsClient

# Load .env variables (if running locally)
load_dotenv()

# Get secrets from Kaggle environment
user_secrets = UserSecretsClient()
langsmith_key = user_secrets.get_secret("LANGSMITH_API_KEY")
api_key = user_secrets.get_secret("OPENAI_API_KEY")

# Set environment variables
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = langsmith_key
os.environ["LANGSMITH_PROJECT"] = "pr-tart-analogue-60"

# LangChain & LangGraph imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, MessageGraph

# Prompt for generating Qiskit code
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart code assistant tasked with coding excellent Qiskit code."
            " Generate the best code possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Prompt for reviewing the generated code
reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a smart code reviewer grading the code. Generate critique and recommendations for the code."
            " Always provide detailed recommendations.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Initialize the LLM
llm = ChatOpenAI(api_key=api_key, model="gpt-4")

# Define chains
generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# Create LangGraph workflow
graph = MessageGraph()

GENERATE = "generate"
REFLECT = "reflect"

def generate_node(state):
    return generation_chain.invoke({"messages": state})

def reflect_node(state):
    return reflection_chain.invoke({"messages": state})

def should_continue(state):
    if len(state) > 4:
        return END
    return REFLECT

# Register nodes and edges
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)
graph.set_entry_point(GENERATE)
graph.add_conditional_edges(GENERATE, should_continue, [REFLECT, END])
graph.add_edge(REFLECT, GENERATE)

# Compile and run
app = graph.compile()

print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# Run the graph with a test question
result = app.invoke(HumanMessage(content="Generate a qiskit code which maps the simple knapsack problem onto quantum computers"))
print(result)
