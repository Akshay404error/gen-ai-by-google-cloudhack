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
import json
import re

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="AI Test Case Generator",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .tech-card {
        background-color: #e8f5e8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2e7d32;
        margin: 0.5rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# AI Prompts for different domains
DOMAIN_PROMPTS = {
    "general": """
    Generate comprehensive test cases for general software functionality.
    Focus on user stories, business logic, and typical software features.
    """,
    "embedded": """
    Generate test cases for embedded systems and hardware interfaces.
    Focus on low-level operations, hardware interactions, real-time constraints,
    and safety-critical functionality. Include error handling for hardware failures.
    """,
    "thermal": """
    Generate test cases for thermal monitoring and management systems.
    Focus on temperature sensors, thermal zones, hardware interfaces,
    error conditions, and safety mechanisms. Include boundary testing
    for temperature ranges and timing constraints.
    """,
    "safety": """
    Generate test cases for safety-critical systems.
    Focus on fault tolerance, error recovery, redundancy,
    and compliance with safety standards like ISO 26262.
    """
}

# Technical patterns for automatic detection
TECHNICAL_PATTERNS = {
    "thermal": ["thermal", "temperature", "sensor", "zone", "°C", "millidegrees", "BPMP", "SOC_THERM"],
    "embedded": ["embedded", "hardware", "driver", "firmware", "real-time", "RTOS", "register"],
    "safety": ["safety", "critical", "fault", "recovery", "redundancy", "ISO26262", "ASIL"],
    "general": ["user", "interface", "business", "application", "web", "mobile"]
}

class TestCase(BaseModel):
    test_case_id: int = Field(..., description="Unique identifier for the test case.")
    test_title: str = Field(..., description="Title of the test case.")
    description: str = Field(..., description="Detailed description of what the test case covers.")
    preconditions: str = Field(..., description="Any setup required before execution.")
    test_steps: List[str] = Field(..., description="Step-by-step execution guide.")
    test_data: str = Field(..., description="Input values required for the test.")
    expected_result: str = Field(..., description="The anticipated outcome.")
    priority: str = Field("Medium", description="Priority level: High, Medium, Low")
    test_type: str = Field("Functional", description="Type of test: Functional, Regression, etc.")
    domain: str = Field("general", description="Domain: general, embedded, thermal, safety")
    comments: str = Field(..., description="Additional notes or observations.")

class OutputSchema(BaseModel):
    test_cases: List[TestCase] = Field(..., description="List of test cases.")

class State(TypedDict):
    test_cases: List[TestCase]
    user_story: str
    domain: str

def detect_domain(user_story: str) -> str:
    """Automatically detect the domain based on technical keywords"""
    user_story_lower = user_story.lower()
    
    for domain, patterns in TECHNICAL_PATTERNS.items():
        if any(pattern.lower() in user_story_lower for pattern in patterns):
            return domain
    
    return "general"

def export_to_excel(test_cases: List[dict]) -> BytesIO:
    """Export test cases to Excel with enhanced formatting"""
    try:
        df = pd.DataFrame(test_cases)
        
        # Create Excel file
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Test Cases')
            
            # Format worksheets
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column in worksheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    worksheet.column_dimensions[column_letter].width = adjusted_width
        
        output.seek(0)
        return output
        
    except Exception as e:
        st.error(f"Excel export failed: {str(e)}")
        return BytesIO()

def generate_thermal_test_cases(user_story: str, count: int = 5) -> List[dict]:
    """Specialized thermal monitoring test case generator"""
    return [
        {
            "test_case_id": 1,
            "test_title": "Verify successful temperature query for thermal zone",
            "description": "Test temperature retrieval for supported thermal zones in SOC_THERM domain",
            "preconditions": "Thermal zone supported, NvThermmonOpen returns success, system in normal state",
            "test_steps": [
                "Open connection to thermal zone using NvThermmonOpen()",
                "Query temperature using NvThermmonGetZoneTemp()",
                "Retrieve two consecutive temperature values",
                "Validate temperature values are within expected range"
            ],
            "test_data": "Valid thermal zone name (e.g., 'CPU-therm'), expected temperature range",
            "expected_result": "API returns success with two valid temperature values differing by acceptable delta",
            "priority": "High",
            "test_type": "Functional",
            "domain": "thermal",
            "comments": "Validates core thermal monitoring functionality"
        },
        {
            "test_case_id": 2,
            "test_title": "Verify timestamp accuracy for temperature queries",
            "description": "Test that timestamps are properly recorded with temperature readings",
            "preconditions": "Thermal zone accessible, system clock synchronized",
            "test_steps": [
                "Open connection to thermal zone",
                "Query temperature multiple times",
                "Capture timestamps for each reading",
                "Calculate time differences between readings"
            ],
            "test_data": "Thermal zone handle, timestamp validation threshold (e.g., 1000 microseconds)",
            "expected_result": "Timestamps are accurate and time differences between readings are within acceptable limits",
            "priority": "Medium",
            "test_type": "Performance",
            "domain": "thermal",
            "comments": "Validates timestamp precision and measurement timing"
        },
        {
            "test_case_id": 3,
            "test_title": "Verify error handling for invalid thermal zones",
            "description": "Test proper error handling when accessing non-existent thermal zones",
            "preconditions": "System running, thermal monitoring service active",
            "test_steps": [
                "Attempt to open connection to invalid thermal zone name",
                "Verify error code returned",
                "Attempt temperature query on invalid handle",
                "Validate error handling behavior"
            ],
            "test_data": "Invalid thermal zone names, expected error codes (NV_THERMMON_ERR_CODE_INVALID_PARAM)",
            "expected_result": "Appropriate error codes returned for invalid operations, system remains stable",
            "priority": "High",
            "test_type": "Negative",
            "domain": "thermal",
            "comments": "Validates robust error handling and system stability"
        },
        {
            "test_case_id": 4,
            "test_title": "Verify temperature resolution preservation",
            "description": "Test that temperature resolution is maintained throughout processing",
            "preconditions": "High-resolution temperature sensor available",
            "test_steps": [
                "Query temperature multiple times",
                "Analyze resolution of returned values",
                "Verify no loss of precision in temperature readings",
                "Compare with sensor specification"
            ],
            "test_data": "Known temperature values, resolution requirements",
            "expected_result": "Temperature resolution preserved as specified in requirements",
            "priority": "Medium",
            "test_type": "Accuracy",
            "domain": "thermal",
            "comments": "Validates data precision and measurement accuracy"
        },
        {
            "test_case_id": 5,
            "test_title": "Verify thermal zone boundary conditions",
            "description": "Test temperature reading at operational boundaries",
            "preconditions": "Thermal zone accessible, temperature control available",
            "test_steps": [
                "Set thermal zone to minimum operational temperature",
                "Query temperature and validate reading",
                "Set thermal zone to maximum operational temperature",
                "Query temperature and validate reading"
            ],
            "test_data": "Boundary temperature values, acceptable tolerance ranges",
            "expected_result": "Temperature readings accurate at boundary conditions, proper handling of extreme values",
            "priority": "High",
            "test_type": "Boundary",
            "domain": "thermal",
            "comments": "Validates system behavior at operational limits"
        }
    ]

def generate_mock_test_cases(user_story: str, domain: str = "general", count: int = 3) -> List[dict]:
    """Enhanced mock test case generator with domain support"""
    if domain == "thermal":
        return generate_thermal_test_cases(user_story, count)
    
    # General test cases for other domains
    return [
        {
            "test_case_id": 1,
            "test_title": f"Functional Test - {user_story[:25]}...",
            "description": f"Validate main functionality: {user_story}",
            "preconditions": "System is available and test environment ready",
            "test_steps": ["Navigate to feature", "Perform primary action", "Verify results"],
            "test_data": "Valid input data, typical usage scenario",
            "expected_result": "Function works as expected without errors",
            "priority": "High",
            "test_type": "Functional",
            "domain": domain,
            "comments": "Primary functionality validation"
        },
        {
            "test_case_id": 2,
            "test_title": f"Error Handling Test - {user_story[:25]}...",
            "description": f"Test error scenarios for: {user_story}",
            "preconditions": "System is available",
            "test_steps": ["Provide invalid input", "Attempt operation", "Check error handling"],
            "test_data": "Invalid data, edge cases",
            "expected_result": "Appropriate error messages displayed",
            "priority": "Medium",
            "test_type": "Negative",
            "domain": domain,
            "comments": "Error scenario validation"
        },
        {
            "test_case_id": 3,
            "test_title": f"Performance Test - {user_story[:25]}...",
            "description": f"Test performance characteristics for: {user_story}",
            "preconditions": "System under normal load",
            "test_steps": ["Execute operation multiple times", "Measure response times", "Check resource usage"],
            "test_data": "Typical workload, performance thresholds",
            "expected_result": "Performance meets specified requirements",
            "priority": "Medium",
            "test_type": "Performance",
            "domain": domain,
            "comments": "Performance validation"
        }
    ]

def create_test_case_generator(domain: str = "general"):
    """Create AI test case generator with domain-specific tuning"""
    try:
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3 if domain == "technical" else 0.2,
            max_tokens=4000,
            timeout=30,
            max_retries=2,
        )
        
        return llm.with_structured_output(OutputSchema)
        
    except Exception as e:
        st.error(f"Failed to initialize AI model: {e}")
        return None

def generate_ai_test_cases(user_story: str, domain: str = "general", count: int = 5) -> List[dict]:
    """Generate test cases using AI with domain awareness"""
    try:
        generator = create_test_case_generator(domain)
        if not generator:
            return generate_mock_test_cases(user_story, domain, count)
        
        prompt = f"""
        {DOMAIN_PROMPTS.get(domain, DOMAIN_PROMPTS['general'])}
        
        User Story/Requirement: {user_story}
        
        Generate {count} comprehensive test cases. For each test case, include:
        - Unique test case ID
        - Descriptive title indicating test purpose
        - Detailed description of what is being tested
        - Clear preconditions and setup requirements
        - Step-by-step test steps (as list)
        - Required test data and inputs
        - Expected results and validation criteria
        - Priority level (High, Medium, Low)
        - Test type (Functional, Negative, Performance, etc.)
        - Domain context ({domain})
        - Additional comments and observations
        
        Focus on {domain}-specific testing aspects and include both positive and negative scenarios.
        """
        
        response = generator.invoke(prompt)
        test_cases = response.test_cases if hasattr(response, 'test_cases') else []
        
        # Convert to list of dictionaries
        test_cases_dicts = []
        for i, test_case in enumerate(test_cases, 1):
            if hasattr(test_case, 'dict'):
                tc_dict = test_case.dict()
            else:
                tc_dict = test_case
            tc_dict['test_case_id'] = i
            test_cases_dicts.append(tc_dict)
        
        return test_cases_dicts[:count]
        
    except Exception as e:
        st.error(f"AI generation failed: {e}")
        return generate_mock_test_cases(user_story, domain, count)

def main():
    st.markdown('<h1 class="main-header">🧪 AI-Powered Test Case Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Generate comprehensive test cases for any domain - from user stories to complex technical requirements")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Domain selection
        domain = st.selectbox(
            "Domain:",
            list(DOMAIN_PROMPTS.keys()),
            help="Select the application domain for better test generation"
        )
        
        # Auto-detect button
        if st.button("🔍 Auto-Detect Domain", help="Automatically detect domain from input"):
            if 'user_story' in st.session_state and st.session_state.user_story:
                detected = detect_domain(st.session_state.user_story)
                st.success(f"Detected domain: {detected}")
                domain = detected
        
        with st.expander("Advanced Options"):
            num_test_cases = st.slider(
                "Number of test cases:",
                min_value=1, max_value=10, value=5
            )
            
            use_ai = st.toggle("Use AI Generation", value=True)
            
            st.info("AI uses GROQ API for intelligent test case generation")
        
        # Technical examples
        st.header("🚀 Technical Examples")
        tech_example = st.selectbox(
            "Load technical example:",
            ["Select example...", "Thermal Monitoring", "Embedded Driver", "Safety System", "API Testing"]
        )
        
        if tech_example != "Select example...":
            examples = {
                "Thermal Monitoring": "When a request to retrieve the temperature of a thermal zone in the SOC_THERM domain is made, the system shall retrieve and return two consecutive temperature values preserving resolution with corresponding timestamps.",
                "Embedded Driver": "The device driver shall handle hardware interrupts and provide proper error codes for invalid operations with timeout mechanisms.",
                "Safety System": "The safety-critical system shall implement redundancy and fault detection with automatic recovery mechanisms.",
                "API Testing": "The REST API shall support CRUD operations with proper authentication, validation, and error handling."
            }
            st.session_state.user_story = examples[tech_example]
            domain = detect_domain(examples[tech_example])

    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_story = st.text_area(
            "📝 **Requirement / User Story:**",
            placeholder="Describe your requirement or user story...",
            height=150,
            key="user_story",
            help="Be specific about functionality, including technical details for better results"
        )
        
        # Show domain detection
        if user_story.strip():
            detected_domain = detect_domain(user_story)
            if detected_domain != "general":
                st.info(f"🎯 **Detected Domain:** {detected_domain.upper()} - Technical requirements detected!")
    
    with col2:
        st.markdown("""
        ### 💡 Technical Tips
        - Include specific error codes
        - Mention hardware interfaces
        - Specify timing constraints
        - Define boundary values
        - Include safety requirements
        """)
        
        if domain == "thermal":
            st.markdown("""
            <div class="tech-card">
            <strong>🌡️ Thermal Testing</strong>
            - Temperature ranges
            - Sensor resolution
            - Timestamp accuracy
            - Error conditions
            - Boundary values
            </div>
            """, unsafe_allow_html=True)

    # Generate button
    if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
        if user_story.strip():
            with st.spinner(f"🧠 Generating {domain} test cases..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("🔍 Analyzing requirements...")
                    progress_bar.progress(20)
                    time.sleep(0.5)
                    
                    status_text.text("🤖 Generating test cases...")
                    progress_bar.progress(50)
                    
                    if use_ai:
                        test_cases = generate_ai_test_cases(user_story, domain, num_test_cases)
                    else:
                        test_cases = generate_mock_test_cases(user_story, domain, num_test_cases)
                    
                    progress_bar.progress(80)
                    
                    excel_file = export_to_excel(test_cases)
                    progress_bar.progress(100)
                    
                    status_text.text("✅ Generation complete!")
                    
                    # Display results
                    st.success(f"🎉 Generated {len(test_cases)} {domain} test cases!")
                    
                    # Show metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Cases", len(test_cases))
                    with col2:
                        st.metric("Domain", domain.upper())
                    with col3:
                        high_prio = sum(1 for tc in test_cases if tc.get('priority') == 'High')
                        st.metric("High Priority", high_prio)
                    
                    # Display test cases
                    st.subheader("📋 Generated Test Cases")
                    
                    for tc in test_cases:
                        with st.expander(f"TC-{tc['test_case_id']}: {tc['test_title']} ({tc['priority']})"):
                            st.write(f"**Description:** {tc['description']}")
                            st.write(f"**Type:** {tc['test_type']} | **Domain:** {tc['domain']}")
                            
                            st.write("**Steps:**")
                            for i, step in enumerate(tc.get('test_steps', []), 1):
                                st.write(f"{i}. {step}")
                            
                            st.write(f"**Expected:** {tc['expected_result']}")
                            st.write(f"**Data:** {tc['test_data']}")
                            st.write(f"**Comments:** {tc['comments']}")
                    
                    # Download options
                    st.subheader("📥 Download Options")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.download_button(
                            label="💾 Download Excel",
                            data=excel_file.getvalue(),
                            file_name=f"test_cases_{domain}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    with col2:
                        json_data = json.dumps(test_cases, indent=2)
                        st.download_button(
                            label="📄 Download JSON",
                            data=json_data,
                            file_name=f"test_cases_{domain}.json",
                            mime="application/json"
                        )
                
                except Exception as e:
                    st.error(f"Error during generation: {str(e)}")
        else:
            st.warning("⚠️ Please enter a requirement or user story")

if __name__ == "__main__":
    main()