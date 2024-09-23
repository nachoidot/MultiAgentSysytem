# tools/regression_tool/select_dependent.py
import openai
import pandas as pd
from typing import Any, Dict

# OpenAI API 키 설정 (환경변수로 설정하는 것을 권장)
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

def select_dependent_variable(data: Dict[str, Any], analysis_purpose: str = "investment analysis") -> str:
    """
    주어진 데이터에서 에이전트가 생각을 통해 종속 변수를 선택합니다.
    
    Parameters:
        data (Dict[str, Any]): 분석할 데이터. 각 키는 변수 이름, 값은 리스트 형태의 데이터.
        analysis_purpose (str): 분석의 목적을 설명하는 문장.
    
    Returns:
        str: 선택된 종속 변수 이름.
    """
    try:
        df = pd.DataFrame(data)
        if df.empty:
            raise ValueError("데이터가 비어 있습니다.")
        
        # 데이터 변수에 대한 요약 생성
        summary = df.describe(include='all').to_string()
        
        # 변수 목록 및 데이터 타입
        variables_info = ""
        for column in df.columns:
            variables_info += f"{column}: {df[column].dtype}, Non-Null Count: {df[column].count()}, Unique Values: {df[column].nunique()}\n"
        
        # LLM에게 종속 변수 선택 요청
        prompt = (
            f"다음은 주어진 데이터셋의 요약입니다:\n\n"
            f"분석 목적: {analysis_purpose}\n\n"
            f"데이터 요약:\n{summary}\n\n"
            f"변수 정보:\n{variables_info}\n\n"
            f"위의 정보를 바탕으로, 가장 적절한 종속 변수를 선택하고 그 이유를 설명해 주세요. 종속 변수의 이름만 반환해 주세요."
        )
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert data scientist."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.3,
        )
        
        selected_variable = response.choices[0].message['content'].strip()
        
        # 선택된 변수 유효성 검사
        if selected_variable not in df.columns:
            raise ValueError(f"선택된 변수 '{selected_variable}'가 데이터셋에 존재하지 않습니다.")
        
        return selected_variable
    except Exception as e:
        raise ValueError(f"종속 변수 선택 중 오류 발생: {str(e)}")
