import os
from dotenv import load_dotenv

load_dotenv()  
langsmith_key = os.environ["LANGSMITH_API_KEY"]
groq_key = os.environ["GROQ_API_KEY"]
tavily_key = os.environ["TAVILY_API_KEY"]

os.environ["LANGSMITH_TRACING"] = os.environ.get("LANGSMITH_TRACING", "true")
os.environ["LANGSMITH_ENDPOINT"] = os.environ.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
os.environ["LANGSMITH_PROJECT"] = os.environ.get("LANGSMITH_PROJECT", "qml-qiskit-patterns")

from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.types import Command, interrupt
from typing import TypedDict, Annotated, List
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import uuid

llm = ChatGroq(model="llama-3.1-8b-instant")


class State(TypedDict):
    problem_statement: str
    generated_code: Annotated[List[str], add_messages]
    human_feedback: Annotated[List[str], add_messages]


def model(state: State): 
    """ Here, we're using the LLM to generate a qiskit python code with human feedback incorporated """

    print("[model] Generating code")
    problem_statement = state["problem_statement"]
    feedback = state["human_feedback"] if "human_feedback" in state else ["No Feedback yet"]


    # Here, we define the prompt 

    prompt = f"""
    Problem Statement: {problem_statement}
    Human Feedback: {feedback[-1] if feedback else "No feedback yet"}
    
    Generate only the Qiskit code (no explanations or comments) that solves the given problem statement. Always include the following imports at the top of the code, regardless of whether all are used:
    
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        from qiskit.circuit.library import ZZFeatureMap
        from qiskit.primitives import StatevectorSampler
        from qiskit_machine_learning.state_fidelities import ComputeUncompute
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
        from qiskit_machine_learning.algorithms import QSVC # for classification
        from qiskit_machine_learning.algorithms import  QSVR # for regression
    

        feature_map = ZZFeatureMap(feature_dimension=, reps=, insert_barriers=True)
        sampler = StatevectorSampler()
        state_fidelity = ComputeUncompute(sampler)
        kernel = FidelityQuantumKernel(fidelity=state_fidelity, feature_map=feature_map)
        qsvc = QSVC(kernel=kernel, tol=, max_iter=)
        qsvr = QSVR(kernel=kernel,tol=,max_iter=,C=)

        Ensure the initializations of feature map, kernel and qsvc/qsvr are ONLY done using this format to prevent syntax error
        
        Do not use np.asnumpy in standard numpy/scikit-learn pipelines
        Dont import Aer from qiskit

        Ensure you print accuracy at the end

    Consider previous human feedback to refine the code.
    """

    response = llm.invoke([
        SystemMessage(content="You are an expert qiskit code generator"), 
        HumanMessage(content=prompt)
    ])

    generated_code = response.content

    print(f"[model_node] Generated code:\n{generated_code}\n")

    return {
       "generated_code": [AIMessage(content=generated_code)] , 
       "human_feedback": feedback
    }


def human_node(state: State): 
    """Human Intervention node - loops back to model unless input is done"""

    print("\n [human_node] awaiting human feedback...")

    generated_code = state["generated_code"]

    # Interrupt to get user feedback

    user_feedback = interrupt(
        {
            "generated_code": generated_code, 
            "message": "Provide feedback or type 'done' to finish"
        }
    )

    print(f"[human_node] Received human feedback: {user_feedback}")

    # If user types "done", transition to EXECUTE node
    if user_feedback.lower() == "done": 
        return Command(update={"human_feedback": state["human_feedback"]}, goto="execute_node")

    # if user types "exit", transition to END node
    elif user_feedback.lower() == "exit" :
        return Command(update={},goto = "end_node")

    # Otherwise, update feedback and return to model for re-generation
    return Command(update={"human_feedback": state["human_feedback"] + [user_feedback]}, goto="model")


import re

def strip_code_fences(code: str) -> str:
    # Matches triple backticks, optional 'python', newline, then captures the code until the next triple backticks
    pattern = r"```(?:python)?\\n([\\s\\S]*?)```"
    # For a real newline, replace '\\n' (backslash-n) with '\n' (newline):
    pattern = r"```(?:python)?\n([\s\S]*?)```"
    match = re.search(pattern, code)
    return match.group(1) if match else code


def execute_node(state: State): 
    """Executes the generated Qiskit code and returns its output or error."""
    import sys
    import io
    import traceback

    generated_code_list = state["generated_code"]
    if not generated_code_list:
        print("[execute_node] No generated code to execute.")
        return {"execution_output": "No code to execute."}

    
    latest_code = generated_code_list[-1]
    code_str = strip_code_fences(
        latest_code.content if hasattr(latest_code, "content") else str(latest_code)
    )



    # Redirect stdout and stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

    try:
        local_vars = {}
        exec(code_str, {}, local_vars)
        output = sys.stdout.getvalue()
        error_output = sys.stderr.getvalue()
        if error_output:
            output += f"\nWarnings:\n{error_output}"
    except Exception:
        output = f"Exception during code execution:\n{traceback.format_exc()}"
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

    print(f"[execute_node] Execution output:\n{output}")
    return {"execution_output": output}


def end_node(state: State): 
    """ Final node """
    print("\n[end_node] Process finished")


graph = StateGraph(State)
graph.add_node("model", model)
graph.add_node("human_node", human_node)
graph.add_node("execute_node", execute_node)
graph.add_node("end_node", end_node)

graph.set_entry_point("model")

# Define the flow

graph.add_edge(START, "model")
graph.add_edge("model", "human_node")
graph.add_edge("execute_node", "human_node")

graph.set_finish_point("end_node")

# Enable Interrupt mechanism
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

thread_config = {"configurable": {
    "thread_id": uuid.uuid4()
}}

problem_statement = input("Enter your Problem Statement: ")
initial_state = {
    "problem_statement": problem_statement,
    "generated_code": [],
    "human_feedback": []
}


for chunk in app.stream(initial_state, config=thread_config):
    for node_id, value in chunk.items():
        #  If we reach an interrupt, continuously ask for human feedback

        if(node_id == "__interrupt__"):
            while True: 
                user_feedback = input("Provide feedback (type 'done' when ready to execute and 'exit' to terminate program): ")

                # Resume the graph execution with the user's feedback
                app.invoke(Command(resume=user_feedback), config=thread_config)

                # Exit loop if user says done
                if user_feedback.lower() == "exit":
                    break

