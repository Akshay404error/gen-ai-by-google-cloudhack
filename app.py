import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the project directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from tester_agent import generate_test_cases, generate_mock_test_cases, TestCase
from export_to_excel import export_to_excel

# Test Data
SAMPLE_USER_STORY = "As a user, I want to login to the system using my email and password."

class TestMockTestCases:
    """Test the mock test case generation fallback"""
    
    def test_generate_mock_test_cases_basic(self):
        """Test mock test case generation with basic user story"""
        test_cases = generate_mock_test_cases(SAMPLE_USER_STORY)
        
        assert isinstance(test_cases, list)
        assert len(test_cases) >= 2
        assert test_cases[0]['test_case_id'] == 1
        assert "Test" in test_cases[0]['test_title']

class TestExcelExport:
    """Test Excel export functionality"""
    
    def test_export_to_excel_success(self):
        """Test successful Excel export"""
        test_cases = generate_mock_test_cases(SAMPLE_USER_STORY)
        excel_file = export_to_excel(test_cases)
        
        assert hasattr(excel_file, 'getvalue')
        
        # Verify Excel content can be read
        df = pd.read_excel(excel_file.getvalue())
        assert len(df) >= 2
        assert 'test_title' in df.columns

class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_mock_generation(self):
        """Test complete flow with mock data"""
        # Generate test cases
        test_cases = generate_mock_test_cases(SAMPLE_USER_STORY)
        
        # Export to Excel
        excel_file = export_to_excel(test_cases)
        
        # Verify results
        assert len(test_cases) > 0
        assert hasattr(excel_file, 'getvalue')
        
        # Read back from Excel to verify data integrity
        df = pd.read_excel(excel_file.getvalue())
        assert len(df) == len(test_cases)

# Run tests with: python -m pytest test_agent.py -v