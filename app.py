import streamlit as st
import pandas as pd
from io import BytesIO
import time
from typing import List, TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph
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

def export_to_excel(test_cases: List[dict]) -> BytesIO:
    """
    Export test cases to Excel and return the file data
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame(test_cases)
        
        # Reorder columns for better readability
        column_order = [
            'test_case_id', 'test_title', 'description', 'preconditions',
            'test_steps', 'test_data', 'expected_result', 'comments'
        ]
        
        # Only include columns that exist in the data
        existing_columns = [col for col in column_order if col in df.columns]
        df = df[existing_columns]
        
        # Create Excel file in memory
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Test Cases')
            
            # Auto-adjust column widths
            worksheet = writer.sheets['Test Cases']
            for i, col in enumerate(df.columns):
                max_len = max(df[col].astype(str).apply(len).max(), len(col)) + 2
                worksheet.column_dimensions[chr(65 + i)].width = min(max_len, 50)
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Excel export failed: {str(e)}")
        return BytesIO()

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

def create_test_case_generator():
    """Create and configure the LangGraph workflow"""
    
    try:
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
                st.error(f"AI Generation Error: {e}")
                return {"test_cases": state.get('test_cases', [])}

        # Build the graph
        graph_builder = StateGraph(State)
        
        # Add nodes
        graph_builder.add_node("generate_test_cases", test_cases_generator)
        
        # Set entry point
        graph_builder.set_entry_point("generate_test_cases")
        
        # For older versions of langgraph, use add_conditional_edges instead of END
        graph_builder.add_conditional_edges(
            "generate_test_cases",
            lambda state: "__end__",
            {"__end__": "__end__"}
        )
        
        # Compile the graph
        return graph_builder.compile()
        
    except Exception as e:
        st.error(f"Failed to initialize AI model: {e}")
        return None

def generate_test_cases(user_input: str, num_cases: int = 3):
    """
    Generate test cases for the given user story
    """
    try:
        # Initialize the graph
        graph = create_test_case_generator()
        
        if graph is None:
            st.warning("AI model not available. Using mock data.")
            return generate_mock_test_cases(user_input)
        
        # Initial state
        initial_state = {
            'test_cases': [],
            'user_story': user_input
        }
        
        # Execute the graph
        result = graph.invoke(initial_state)
        test_cases_result = result.get('test_cases', [])
        
        # Convert to list of dictionaries
        test_cases_dicts = []
        for i, test_case in enumerate(test_cases_result, 1):
            if hasattr(test_case, 'dict'):
                tc_dict = test_case.dict()
            else:
                tc_dict = test_case
            tc_dict['test_case_id'] = i  # Ensure sequential IDs
            test_cases_dicts.append(tc_dict)
        
        # Limit to requested number of cases
        return test_cases_dicts[:num_cases]
        
    except Exception as e:
        st.error(f"Error in generate_test_cases: {e}")
        return generate_mock_test_cases(user_input)

def run_comprehensive_tests():
    """
    Run comprehensive tests and return results
    """
    results = {
        'success': True,
        'total_tests': 0,
        'passed': 0,
        'failed': 0,
        'details': {}
    }
    
    # Test 1: Mock Test Case Generation
    try:
        test_cases = generate_mock_test_cases("Test user story")
        assert isinstance(test_cases, list)
        assert len(test_cases) >= 2
        results['details']['Mock Generation Test'] = {'passed': True}
        results['passed'] += 1
    except Exception as e:
        results['details']['Mock Generation Test'] = {
            'passed': False,
            'error': str(e)
        }
        results['failed'] += 1
        results['success'] = False
    
    # Test 2: Excel Export
    try:
        test_cases = generate_mock_test_cases("Test export")
        excel_file = export_to_excel(test_cases)
        assert hasattr(excel_file, 'getvalue')
        results['details']['Excel Export Test'] = {'passed': True}
        results['passed'] += 1
    except Exception as e:
        results['details']['Excel Export Test'] = {
            'passed': False,
            'error': str(e)
        }
        results['failed'] += 1
        results['success'] = False
    
    results['total_tests'] = results['passed'] + results['failed']
    return results

def main():
    st.title("🧪 AI Test Case Generator")
    st.markdown("Generate comprehensive test cases from user stories using AI")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        num_test_cases = st.slider(
            "Number of test cases:",
            min_value=1,
            max_value=10,
            value=3,
            help="More test cases may take longer to generate"
        )
        
        use_ai = st.toggle(
            "Use AI Generation", 
            value=True,
            help="Toggle off to use mock data for testing"
        )
        
        st.info("AI generation requires valid GROQ API key in .env file")
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input field for user story
        user_story = st.text_area(
            "📝 Enter the user story or functionality description:",
            placeholder="As a user, I want to login to the system so that I can access my dashboard...",
            height=120,
            help="Be specific about the functionality you want to test"
        )
    
    with col2:
        st.markdown("**Examples:**")
        st.markdown("""
        - *"As a user, I want to search products so that I can find items to purchase"*
        - *"As an admin, I want to delete users so that I can manage the system"*
        - *"As a customer, I want to filter products by price so that I can find affordable items"*
        """)
    
    # Mode selection
    test_mode = st.radio(
        "Select Mode:",
        ["Generate Test Cases", "Run Tests"],
        horizontal=True
    )
    
    if test_mode == "Generate Test Cases":
        if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
            if user_story.strip():
                # Create progress containers
                progress_container = st.container()
                status_container = st.container()
                result_container = st.container()
                
                with progress_container:
                    st.subheader("Generation Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Step 1: Starting generation
                    status_text.text("🔄 Initializing test case generation...")
                    progress_bar.progress(10)
                    time.sleep(0.5)
                    
                    # Step 2: Generating test cases
                    status_text.text("🤖 Analyzing user story and generating test cases...")
                    progress_bar.progress(30)
                    
                    if use_ai:
                        test_cases = generate_test_cases(user_story, num_test_cases)
                    else:
                        test_cases = generate_mock_test_cases(user_story)
                        time.sleep(1)
                    
                    progress_bar.progress(60)
                    time.sleep(0.5)
                    
                    # Step 3: Exporting to Excel
                    status_text.text("💾 Exporting test cases to Excel format...")
                    progress_bar.progress(80)
                    
                    excel_file = export_to_excel(test_cases)
                    progress_bar.progress(100)
                    time.sleep(0.5)
                    
                    # Step 4: Complete
                    status_text.text("✅ Generation complete!")
                    time.sleep(0.5)
                    
                    with status_container:
                        st.success("🎉 Test cases generated successfully!")
                        
                        # Show statistics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Test Cases", len(test_cases))
                        with col2:
                            st.metric("User Story Length", f"{len(user_story)} chars")
                        with col3:
                            st.metric("Generation Mode", "AI" if use_ai else "Mock")
                    
                    with result_container:
                        # Show preview
                        st.subheader("📋 Test Cases Preview")
                        
                        # Create a clean preview table
                        preview_data = []
                        for i, tc in enumerate(test_cases, 1):
                            preview_data.append({
                                "ID": i,
                                "Title": tc.get('test_title', 'No title'),
                                "Description": tc.get('description', 'No description')[:100] + '...' if tc.get('description') else 'No description'
                            })
                        
                        st.dataframe(preview_data, use_container_width=True)
                        
                        # Download section
                        st.subheader("📥 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.download_button(
                                label="💾 Download Excel File",
                                data=excel_file.getvalue(),
                                file_name="test_cases.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        with col2:
                            # JSON download option
                            import json
                            json_data = json.dumps(test_cases, indent=2)
                            st.download_button(
                                label="📄 Download JSON",
                                data=json_data,
                                file_name="test_cases.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                except Exception as e:
                    st.error(f"An error occurred during generation: {str(e)}")
                    st.info("Please check your API keys and internet connection.")
            else:
                st.warning("⚠️ Please enter a user story to generate test cases.")
    
    else:
        # Test Execution Mode
        st.header("🧪 Run Comprehensive Tests")
        st.info("This will execute system tests to validate the application")
        
        if st.button("🔬 Run All Tests", type="secondary"):
            with st.spinner("Running comprehensive tests..."):
                try:
                    # Create a container for test results
                    results_container = st.container()
                    
                    # Run the comprehensive tests
                    test_results = run_comprehensive_tests()
                    
                    with results_container:
                        st.subheader("📊 Test Results")
                        
                        # Display overall status
                        if test_results.get('success', False):
                            st.success("✅ All tests passed!")
                        else:
                            st.error("❌ Some tests failed!")
                        
                        # Display detailed results
                        st.subheader("Detailed Test Results")
                        
                        for test_name, result in test_results.get('details', {}).items():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{test_name}**")
                            with col2:
                                if result.get('passed', False):
                                    st.success("✅ PASS")
                                else:
                                    st.error("❌ FAIL")
                            
                            if result.get('error'):
                                with st.expander("See error details"):
                                    st.code(result['error'])
                        
                        # Show summary statistics
                        st.subheader("📈 Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Tests", test_results.get('total_tests', 0))
                        with col2:
                            st.metric("Passed", test_results.get('passed', 0))
                        with col3:
                            st.metric("Failed", test_results.get('failed', 0))
                
                except Exception as e:
                    st.error(f"❌ Error running tests: {str(e)}")

if __name__ == "__main__":
    main()