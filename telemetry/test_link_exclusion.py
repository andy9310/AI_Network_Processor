#!/usr/bin/env python3
"""
Test script for link exclusion functionality
Tests the parse_protected_links and filter_links_to_close functions
"""

import re

def parse_protected_links(message: str) -> list:
    """Parse user message to extract protected links from brackets []
    
    Args:
        message: User message that may contain protected links like [S1-S4]
        
    Returns:
        list: List of protected link names (including bidirectional pairs)
    """
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
    """Filter out protected links from predicted links to close
    
    Args:
        predicted_links: Links predicted by the model to be closed
        protected_links: Links that user wants to keep open
        
    Returns:
        list: Filtered links excluding protected ones
    """
    if not protected_links:
        return predicted_links
    
    filtered_links = [link for link in predicted_links if link not in protected_links]
    
    if len(filtered_links) != len(predicted_links):
        excluded = [link for link in predicted_links if link in protected_links]
        print(f"🔒 Excluded protected links: {excluded}")
        print(f"📋 Filtered links to close: {filtered_links}")
    
    return filtered_links

def test_link_exclusion():
    """Test the link exclusion functionality with various scenarios"""
    
    print("🧪 Testing Link Exclusion Functionality\n")
    
    # Test cases
    test_cases = [
        {
            "message": "在不關閉[S1-S4]的情況下進行節能",
            "predicted_links": ["S1-S2", "S1-S4", "S3-S4", "S5-S9"],
            "description": "Chinese message with single protected link"
        },
        {
            "message": "Please optimize energy while keeping [S1-S2, S3-S4] active",
            "predicted_links": ["S1-S2", "S1-S4", "S3-S4", "S5-S9"],
            "description": "English message with multiple protected links (comma separated)"
        },
        {
            "message": "Energy saving without closing [S1-S2 S3-S4 S5-S9]",
            "predicted_links": ["S1-S2", "S1-S4", "S3-S4", "S5-S9"],
            "description": "Multiple protected links (space separated)"
        },
        {
            "message": "Optimize network performance",
            "predicted_links": ["S1-S2", "S1-S4", "S3-S4", "S5-S9"],
            "description": "No protected links specified"
        },
        {
            "message": "Keep [S1-S2] and [S3-S4] open while optimizing [S5-S9]",
            "predicted_links": ["S1-S2", "S1-S4", "S3-S4", "S5-S9"],
            "description": "Multiple bracket groups with mixed intentions"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}: {test_case['description']}")
        print(f"Message: {test_case['message']}")
        print(f"Predicted links: {test_case['predicted_links']}")
        
        # Parse protected links
        protected_links = parse_protected_links(test_case['message'])
        print(f"Protected links parsed: {protected_links}")
        
        # Filter links
        filtered_links = filter_links_to_close(test_case['predicted_links'], protected_links)
        print(f"Final filtered links: {filtered_links}")
        
        print("-" * 60)
        print()

if __name__ == "__main__":
    test_link_exclusion()
