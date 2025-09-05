# test_fix.py
from tester_agent import generate_test_cases

# Test the fixed function
user_story = "As a user, I want to login to the system so that I can access my dashboard."
test_cases = generate_test_cases(user_story)

print(f"Generated {len(test_cases)} test cases:")
for i, tc in enumerate(test_cases, 1):
    print(f"\n{i}. {tc.get('test_title', 'No Title')}")
    print(f"   Description: {tc.get('description', 'No description')[:100]}...")