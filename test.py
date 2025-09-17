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
</style>
""", unsafe_allow_html=True)

# AI Prompts for different domains
DOMAIN_PROMPTS = {
    "general": "Generate comprehensive test cases for general software functionality.",
    "embedded": "Generate test cases for embedded systems with hardware interfaces and real-time constraints.",
    "thermal": "Generate test cases for thermal monitoring systems with temperature sensors and error handling.",
    "safety": "Generate test cases for safety-critical systems with fault tolerance and redundancy."
}

# Technical patterns for automatic detection
TECHNICAL_PATTERNS = {
    "thermal": ["thermal", "temperature", "sensor", "zone", "°C", "millidegrees"],
    "embedded": ["embedded", "hardware", "driver", "firmware", "real-time"],
    "safety": ["safety", "critical", "fault", "recovery", "redundancy"],
    "general": ["user", "interface", "business", "application"]
}

def detect_domain(user_story: str) -> str:
    """Automatically detect the domain based on technical keywords"""
    user_story_lower = user_story.lower()
    
    for domain, patterns in TECHNICAL_PATTERNS.items():
        if any(pattern.lower() in user_story_lower for pattern in patterns):
            return domain
    
    return "general"

def export_to_excel(test_cases: List[dict]) -> BytesIO:
    """Export test cases to Excel"""
    try:
        df = pd.DataFrame(test_cases)
        
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Test Cases')
            
            # Format columns
            worksheet = writer.sheets['Test Cases']
            for column in worksheet.columns:
                max_length = max(len(str(cell.value)) for cell in column)
                worksheet.column_dimensions[column[0].column_letter].width = min(max_length + 2, 50)
        
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
            "description": "Test temperature retrieval for supported thermal zones",
            "preconditions": "Thermal zone supported, system in normal state",
            "test_steps": "1. Open connection to thermal zone\n2. Query temperature\n3. Retrieve temperature values\n4. Validate temperature range",
            "test_data": "Valid thermal zone name, expected temperature range",
            "expected_result": "API returns success with valid temperature values",
            "priority": "High",
            "test_type": "Functional",
            "domain": "thermal",
            "comments": "Validates core thermal monitoring functionality"
        },
        {
            "test_case_id": 2,
            "test_title": "Verify timestamp accuracy for temperature queries",
            "description": "Test that timestamps are properly recorded",
            "preconditions": "Thermal zone accessible, system clock synchronized",
            "test_steps": "1. Open connection\n2. Query temperature multiple times\n3. Capture timestamps\n4. Calculate time differences",
            "test_data": "Thermal zone handle, timestamp validation threshold",
            "expected_result": "Timestamps are accurate and within acceptable limits",
            "priority": "Medium",
            "test_type": "Performance",
            "domain": "thermal",
            "comments": "Validates timestamp precision"
        },
        {
            "test_case_id": 3,
            "test_title": "Verify error handling for invalid thermal zones",
            "description": "Test proper error handling for non-existent zones",
            "preconditions": "System running, thermal service active",
            "test_steps": "1. Attempt to open invalid thermal zone\n2. Verify error code\n3. Attempt temperature query\n4. Validate error handling",
            "test_data": "Invalid thermal zone names, expected error codes",
            "expected_result": "Appropriate error codes returned, system remains stable",
            "priority": "High",
            "test_type": "Negative",
            "domain": "thermal",
            "comments": "Validates robust error handling"
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
            "test_steps": "1. Navigate to feature\n2. Perform primary action\n3. Verify results",
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
            "test_steps": "1. Provide invalid input\n2. Attempt operation\n3. Check error handling",
            "test_data": "Invalid data, edge cases",
            "expected_result": "Appropriate error messages displayed",
            "priority": "Medium",
            "test_type": "Negative",
            "domain": domain,
            "comments": "Error scenario validation"
        }
    ]

def generate_simple_ai_test_cases(user_story: str, domain: str = "general", count: int = 5) -> List[dict]:
    """Simplified AI test case generation without structured output"""
    try:
        # Initialize the LLM
        llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=2000,
            timeout=30,
            max_retries=2,
        )
        
        prompt = f"""
        Generate {count} test cases for the following requirement. Focus on {domain} testing.
        
        REQUIREMENT: {user_story}
        
        For each test case, provide:
        - test_case_id: sequential number
        - test_title: descriptive title
        - description: what is being tested
        - preconditions: setup requirements
        - test_steps: step-by-step instructions
        - test_data: required inputs
        - expected_result: expected outcome
        - priority: High/Medium/Low
        - test_type: Functional/Negative/Performance/etc.
        - domain: {domain}
        - comments: additional notes
        
        Return only valid JSON format with a list of test cases.
        """
        
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Try to extract JSON from response
        try:
            # Look for JSON pattern in the response
            json_match = re.search(r'\[\s*\{.*\}\s*\]', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                test_cases = json.loads(json_str)
                
                # Ensure proper format
                for i, tc in enumerate(test_cases, 1):
                    if 'test_case_id' not in tc:
                        tc['test_case_id'] = i
                    if 'domain' not in tc:
                        tc['domain'] = domain
                
                return test_cases[:count]
        except json.JSONDecodeError:
            st.warning("AI response format issue. Using mock data.")
        
        return generate_mock_test_cases(user_story, domain, count)
        
    except Exception as e:
        st.error(f"AI generation failed: {e}")
        return generate_mock_test_cases(user_story, domain, count)

def main():
    st.markdown('<h1 class="main-header">🧪 AI Test Case Generator</h1>', unsafe_allow_html=True)
    st.markdown("### Generate comprehensive test cases for any domain")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Domain selection
        domain = st.selectbox(
            "Domain:",
            list(DOMAIN_PROMPTS.keys()),
            help="Select the application domain"
        )
        
        # Auto-detect button
        if st.button("🔍 Auto-Detect Domain"):
            if 'user_story' in st.session_state and st.session_state.user_story:
                detected = detect_domain(st.session_state.user_story)
                st.success(f"Detected domain: {detected}")
                domain = detected
        
        with st.expander("Advanced Options"):
            num_test_cases = st.slider(
                "Number of test cases:",
                min_value=1, max_value=10, value=3
            )
            
            use_ai = st.toggle("Use AI Generation", value=True)
        
        # Technical examples
        st.header("🚀 Technical Examples")
        tech_example = st.selectbox(
            "Load example:",
            ["Select example...", "Thermal Monitoring", "Safety System", "API Testing"]
        )
        
        if tech_example != "Select example...":
            examples = {
                "Thermal Monitoring": "Retrieve temperature of thermal zone with timestamps and proper error handling",
                "Safety System": "Implement redundancy and fault detection with automatic recovery mechanisms",
                "API Testing": "CRUD operations with authentication, validation, and error handling"
            }
            st.session_state.user_story = examples[tech_example]
            domain = detect_domain(examples[tech_example])

    # Main content
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_story = st.text_area(
            "📝 **Requirement / User Story:**",
            placeholder="Describe your requirement or user story...",
            height=120,
            key="user_story",
            help="Be specific about functionality"
        )
        
        # Show domain detection
        if user_story.strip():
            detected_domain = detect_domain(user_story)
            if detected_domain != "general":
                st.info(f"🎯 **Detected Domain:** {detected_domain.upper()}")
    
    with col2:
        st.markdown("""
        ### 💡 Tips
        - Include specific details
        - Mention error conditions
        - Specify constraints
        - Define expected outcomes
        """)

    # Generate button
    if st.button("🚀 Generate Test Cases", type="primary", use_container_width=True):
        if user_story.strip():
            with st.spinner(f"Generating {domain} test cases..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    status_text.text("Analyzing requirements...")
                    progress_bar.progress(30)
                    
                    if use_ai:
                        test_cases = generate_simple_ai_test_cases(user_story, domain, num_test_cases)
                    else:
                        test_cases = generate_mock_test_cases(user_story, domain, num_test_cases)
                    
                    progress_bar.progress(70)
                    
                    excel_file = export_to_excel(test_cases)
                    progress_bar.progress(100)
                    
                    status_text.text("Generation complete!")
                    
                    # Display results
                    st.success(f"🎉 Generated {len(test_cases)} test cases!")
                    
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
                            st.write(f"**Preconditions:** {tc['preconditions']}")
                            st.write(f"**Steps:** {tc['test_steps']}")
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
                    st.info("Using mock data as fallback")
                    test_cases = generate_mock_test_cases(user_story, domain, num_test_cases)
                    
                    # Display fallback results
                    st.warning("⚠️ Using mock data due to generation issues")
                    for tc in test_cases:
                        with st.expander(f"TC-{tc['test_case_id']}: {tc['test_title']}"):
                            st.write(f"**Description:** {tc['description']}")
                            st.write(f"**Steps:** {tc['test_steps']}")
        else:
            st.warning("⚠️ Please enter a requirement or user story")

if __name__ == "__main__":
    main()