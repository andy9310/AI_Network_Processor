#!/usr/bin/env python3
"""
Test script for specific user cases
"""

import re

def parse_protected_links(message: str) -> list:
    """Parse user message to extract protected links from brackets []"""
    if not message:
        return []
    
    # Find all content within square brackets
    bracket_matches = re.findall(r'\[([^\]]+)\]', message)
    protected_links = []
    
    for match in bracket_matches:
        # Split by comma or space to handle multiple links in one bracket
        links = re.split(r'[,\s]+', match.strip())
        for link in links:
            link = link.strip()
            if link and '-' in link:  # Basic validation for link format
                protected_links.append(link)
                
                # Add bidirectional link (S6-S15 -> also protect S15-S6)
                parts = link.split('-')
                if len(parts) == 2:
                    reverse_link = f"{parts[1]}-{parts[0]}"
                    if reverse_link not in protected_links:
                        protected_links.append(reverse_link)
    
    return protected_links

def filter_links_to_close(predicted_links: list, protected_links: list) -> list:
    """Filter out protected links from predicted links to close"""
    if not protected_links:
        return predicted_links
    
    filtered_links = [link for link in predicted_links if link not in protected_links]
    
    if len(filtered_links) != len(predicted_links):
        excluded = [link for link in predicted_links if link in protected_links]
        print(f"🔒 Excluded protected links: {excluded}")
        print(f"📋 Filtered links to close: {filtered_links}")
    
    return filtered_links

def test_user_specific_cases():
    """Test the specific user cases"""
    
    print("🧪 Testing User-Specific Cases\n")
    
    # Test cases from user
    test_cases = [
        {
            "message": "在不關閉[S1-S4,S1-S9]的情況下，進行節能",
            "predicted_links": ["S1-S2", "S1-S4", "S1-S9", "S3-S4", "S4-S1", "S9-S1"],
            "description": "Multiple links in one bracket (comma separated)"
        },
        {
            "message": "在不關閉[S1-S4,S4-S1]的情況下，進行節能",
            "predicted_links": ["S1-S2", "S1-S4", "S1-S9", "S3-S4", "S4-S1", "S9-S1"],
            "description": "Bidirectional links explicitly mentioned"
        },
        {
            "message": "在不關閉[S6-S15]的情況下，進行節能",
            "predicted_links": ["S1-S4", "S4-S1", "S9-S1", "S6-S4", "S11-S4", "S6-S15", "S15-S6", "S10-S15", "S15-S10", "S16-S10", "S17-S10"],
            "description": "Original user case - single link with bidirectional protection"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Message: {test_case['message']}")
        print(f"Predicted links: {test_case['predicted_links']}")
        
        # Parse protected links
        protected_links = parse_protected_links(test_case['message'])
        print(f"Protected links parsed (including bidirectional): {protected_links}")
        
        # Filter links
        filtered_links = filter_links_to_close(test_case['predicted_links'], protected_links)
        print(f"Final filtered links: {filtered_links}")
        
        print("-" * 80)
        print()

if __name__ == "__main__":
    test_user_specific_cases()
