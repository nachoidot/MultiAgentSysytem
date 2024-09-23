# tools/regression_tool/ridge.py

import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import Any, Dict, float

def ridge_tool(data: Dict[str, Any], alpha: float = 1.0) -> str:
    """
    Ridge 회귀 분석을 수행합니다.
    
    Parameters:
        data (Dict[str, Any]): 분석할 데이터. 각 키는 변수 이름, 값은 리스트 형태의 데이터.
        alpha (float, optional): 정규화 강도. 기본값은 1.0.
    
    Returns:
        str: 회귀 분석 결과 요약.
    """
    try:
        df = pd.DataFrame(data)
        if df.empty:
            return "데이터가 비어 있습니다."
        
        # 종속 변수 선택 (가장 높은 상관관계를 가진 변수)
        corr_matrix = df.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        max_corr = upper_tri.unstack().dropna().sort_values(ascending=False)
        
        if max_corr.empty:
            return "상관관계를 분석할 수 없습니다."
        
        dependent_var = max_corr.idxmax()[0]
        independent_vars = [var for var in df.columns if var != dependent_var]
        
        if not independent_vars:
            return "독립 변수가 없습니다."
        
        X = df[independent_vars]
        X = sm.add_constant(X)  # 상수항 추가
        y = df[dependent_var]
        
        # Ridge 회귀 모델 적합
        model = sm.OLS(y, X)
        ridge = model.fit_regularized(method='elastic_net', alpha=alpha, L1_wt=0)
        
        return ridge.summary().as_text()
    except Exception as e:
        return f"Ridge 회귀 분석 중 오류 발생: {str(e)}"
