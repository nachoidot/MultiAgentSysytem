# tools/regression_tool/glm.py

import statsmodels.api as sm
import pandas as pd
import numpy as np
from typing import Any, Dict

def glm_tool(data: Dict[str, Any], family: str = 'Gaussian') -> str:
    """
    Generalized Linear Models (GLM) 회귀 분석을 수행합니다.
    
    Parameters:
        data (Dict[str, Any]): 분석할 데이터. 각 키는 변수 이름, 값은 리스트 형태의 데이터.
        family (str, optional): 사용될 가족 분포 (예: 'Gaussian', 'Binomial', 'Poisson'). 기본값은 'Gaussian'.
    
    Returns:
        str: 회귀 분석 결과 요약.
    """
    try:
        df = pd.DataFrame(data)
        if df.empty:
            return "데이터가 비어 있습니다."
        
        # 상관관계 분석을 통해 종속 변수 선택 (가장 높은 상관관계를 가진 변수)
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
        
        family = family.lower()
        if family == 'gaussian':
            family_obj = sm.families.Gaussian()
        elif family == 'binomial':
            family_obj = sm.families.Binomial()
        elif family == 'poisson':
            family_obj = sm.families.Poisson()
        else:
            return f"지원되지 않는 가족 분포: {family}. 'Gaussian', 'Binomial', 'Poisson' 중 하나를 선택하세요."
        
        model = sm.GLM(y, X, family=family_obj)
        results = model.fit()
        
        return results.summary().as_text()
    except Exception as e:
        return f"GLM 분석 중 오류 발생: {str(e)}"
