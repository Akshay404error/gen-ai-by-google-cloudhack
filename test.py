from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
import os

# Load environment variables
load_dotenv()

# AI Prompt
GENERATOR_PROMPT = """
You are an expert software tester. Analyse the given user story in depth and generate a comprehensive set of test cases,
including functional, edge, and boundary cases, to ensure complete test coverage of the functionality.

Generate 3-5 detailed test cases for the provided user story. Each test case should include:
- Clear and descriptive title
- Detailed description of what is being tested
- Necessary preconditions
- Step-by-step test steps
- Required test data
- Expected results
- Any additional comments

Focus on both positive and negative test scenarios.
"""

class TestCase(BaseModel):
    test_case_id: int = Field(..., description="Unique identifier for the test case.")
    test_title: str = Field(..., description="Title of the test case.")
    description: str = Field(..., description="Detailed description of what the test case covers.")
    preconditions: str = Field(..., description="Any setup required before execution.")
    test_steps: str = Field(..., description="Step-by-step execution guide.")
    test_data: str = Field(..., description="Input values required for the test.")
    expected_result: str = Field(..., description="The anticipated outcome.")
    comments: str = Field(..., description="Additional notes or observations.")

class OutputSchema(BaseModel):
    test_cases: List[TestCase] = Field(..., description="List of test cases.")

class State(TypedDict):
    test_cases: List[TestCase]
    user_story: str

def create_test_case_generator():
    """Create and configure the LangGraph workflow"""
    
    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Create structured output LLM
    llm_with_structured_output = llm.with_structured_output(OutputSchema)

    def test_cases_generator(state: State):
        """Node to generate test cases"""
        prompt = f"{GENERATOR_PROMPT}\n\nHere is the user story:\n\n{state['user_story']}"
        
        try:
            response = llm_with_structured_output.invoke(prompt)
            # Extract test cases from the structured output
            new_test_cases = response.test_cases if hasattr(response, 'test_cases') else []
            
            # Update the state with new test cases
            current_test_cases = state.get('test_cases', [])
            updated_test_cases = current_test_cases + new_test_cases
            
            return {"test_cases": updated_test_cases}
            
        except Exception as e:
            print(f"Error in test case generation: {e}")
            # Return current state without changes on error
            return {"test_cases": state.get('test_cases', [])}

    # Build the graph
    graph_builder = StateGraph(State)
    
    # Add nodes
    graph_builder.add_node("generate_test_cases", test_cases_generator)
    
    # Set entry point
    graph_builder.set_entry_point("generate_test_cases")
    
    # Set end point (we only run once)
    graph_builder.add_edge("generate_test_cases", END)
    
    # Compile the graph
    return graph_builder.compile()

def generate_test_cases(user_input: str):
    """
    Generate test cases for the given user story
    
    Args:
        user_input: The user story to generate test cases for
    
    Returns:
        List of test case dictionaries
    """
    try:
        # Initialize the graph
        graph = create_test_case_generator()
        
        # Initial state
        initial_state = {
            'test_cases': [],
            'user_story': user_input
        }
        
        # Stream the execution
        test_cases_result = []
        for event in graph.stream(initial_state):
            for value in event.values():
                if 'test_cases' in value:
                    test_cases_result = value['test_cases']
        
        # Convert to list of dictionaries
        test_cases_dicts = []
        for i, test_case in enumerate(test_cases_result, 1):
            tc_dict = test_case.dict() if hasattr(test_case, 'dict') else test_case
            tc_dict['test_case_id'] = i  # Ensure sequential IDs
            test_cases_dicts.append(tc_dict)
        
        return test_cases_dicts
        
    except Exception as e:
        print(f"Error in generate_test_cases: {e}")
        # Fallback to mock generation
        return generate_mock_test_cases(user_input)

def generate_mock_test_cases(user_story: str) -> List[dict]:
    """Fallback mock test case generator"""
    return [
        {
            "test_case_id": 1,
            "test_title": f"Functional Test - {user_story[:30]}...",
            "description": f"Test the main functionality described in: {user_story}",
            "preconditions": "System is available and user has appropriate access rights",
            "test_steps": "1. Navigate to the feature\n2. Perform the main action\n3. Verify results",
            "test_data": "Sample valid input data",
            "expected_result": "Feature works as expected without errors",
            "comments": "Primary functionality test"
        },
        {
            "test_case_id": 2,
            "test_title": f"Error Handling Test - {user_story[:30]}...",
            "description": f"Test error scenarios for: {user_story}",
            "preconditions": "System is available",
            "test_steps": "1. Provide invalid input\n2. Attempt to execute\n3. Check error handling",
            "test_data": "Invalid or edge case input values",
            "expected_result": "Appropriate error messages are displayed",
            "comments": "Error scenario validation"
        },
        {
            "test_case_id": 3,
            "test_title": f"Boundary Test - {user_story[:30]}...",
            "description": f"Test boundary conditions for: {user_story}",
            "preconditions": "System is available",
            "test_steps": "1. Test minimum values\n2. Test maximum values\n3. Test edge cases",
            "test_data": "Boundary value inputs",
            "expected_result": "System handles boundary values correctly",
            "comments": "Boundary condition validation"
        }
    ]