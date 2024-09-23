# tests/__init__.py

import pytest
import os

def load_test_data(file_path):
    """
    테스트 데이터 파일을 로드하는 공용 함수입니다.
    
    Parameters:
        file_path (str): 데이터 파일의 경로.
    
    Returns:
        dict: 로드된 데이터.
    """
    import json
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

@pytest.fixture(scope="session")
def sample_data():
    """
    전체 테스트 세션 동안 사용되는 샘플 데이터를 제공합니다.
    
    Returns:
        dict: 샘플 데이터.
    """
    return {
        "Sales": [100, 150, 200, 250, 300],
        "Advertising": [10, 15, 20, 25, 30],
        "Price": [20, 18, 17, 16, 15],
        "Competition": [5, 3, 6, 4, 2]
    }
