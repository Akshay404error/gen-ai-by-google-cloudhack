import streamlit as st
from export_to_excel import export_to_excel
from tester_agent import generate_test_cases

def main():
    st.title("AI Test Case Generator")
    
    # Input field for user story
    user_story = st.text_area(
        "Enter the user story or functionality description:",
        placeholder="As a user, I want to login to the system so that I can access my dashboard...",
        height=100,
        help="Be specific about the functionality you want to test"
    )
    
    # Optional: Let user choose number of test cases
    num_test_cases = st.slider(
        "Number of test cases to generate:",
        min_value=1,
        max_value=10,
        value=5,
        help="More test cases may take longer to generate"
    )
    
    if st.button("Generate Test Cases", type="primary"):
        if user_story.strip():
            with st.spinner(f"Generating {num_test_cases} test cases..."):
                try:
                    # Generate test cases
                    test_cases = generate_test_cases(user_story, num_test_cases)
                    
                    if test_cases:
                        # Export to Excel
                        excel_file = export_to_excel(test_cases)
                        
                        # Display success message
                        st.success("🎉 Test cases generated successfully!")
                        
                        # Show preview of test cases
                        st.subheader("Test Cases Preview")
                        
                        # Create a simplified preview
                        preview_data = []
                        for tc in test_cases:
                            preview_data.append({
                                "ID": tc.get('test_case_id', 'N/A'),
                                "Title": tc.get('test_title', 'No title'),
                                "Description": tc.get('description', 'No description')[:100] + '...' if tc.get('description') else 'No description'
                            })
                        
                        st.table(preview_data)
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Excel File",
                            data=excel_file.getvalue(),
                            file_name="test_cases.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            help="Download all test cases in Excel format"
                        )
                    else:
                        st.warning("No test cases were generated. Please try again with a different user story.")
                        
                except Exception as e:
                    st.error(f"An error occurred during generation: {str(e)}")
                    st.info("Please check your API keys and internet connection.")
        else:
            st.warning("⚠️ Please enter a user story to generate test cases.")

if __name__ == "__main__":
    main()