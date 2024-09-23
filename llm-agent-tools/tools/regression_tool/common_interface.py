# tools/regression_tool/common_interface.py

from tools.regression_tool.select_dependent import select_dependent_variable
from tools.regression_tool import (
    ols_tool,
    wls_tool,
    gls_tool,
    glm_tool,
    poisson_tool,
    logit_tool,
    ridge_tool,
    lasso_tool
)
from typing import Any, Dict

def analyze_regression(data: Dict[str, Any], regression_type: str, analysis_purpose: str = "general analysis", **kwargs) -> str:
    """
    종속 변수를 자동으로 선택하고, 지정된 회귀 분석을 수행합니다.
    
    Parameters:
        data (Dict[str, Any]): 분석할 데이터.
        regression_type (str): 수행할 회귀 분석 유형 (e.g., 'OLS', 'GLM').
        analysis_purpose (str): 분석의 목적을 설명하는 문장.
        **kwargs: 회귀 분석에 필요한 추가 매개변수.
    
    Returns:
        str: 회귀 분석 결과 요약.
    """
    try:
        # 종속 변수 선택
        dependent_var = select_dependent_variable(data, analysis_purpose=analysis_purpose)
        print(f"Selected Dependent Variable: {dependent_var}")
        
        # 종속 변수 제외한 독립 변수 설정
        independent_vars = {var: data[var] for var in data if var != dependent_var}
        prepared_data = {dependent_var: data[dependent_var]}
        prepared_data.update(independent_vars)
        
        # 회귀 분석 수행
        if regression_type.upper() == 'OLS':
            return ols_tool(prepared_data)
        elif regression_type.upper() == 'WLS':
            weights = kwargs.get('weights', None)
            if weights is None:
                return "WLS 분석을 수행하려면 'weights' 매개변수가 필요합니다."
            return wls_tool(prepared_data, weights)
        elif regression_type.upper() == 'GLS':
            sigma = kwargs.get('sigma', None)
            return gls_tool(prepared_data, sigma=sigma)
        elif regression_type.upper() == 'GLM':
            family = kwargs.get('family', 'Gaussian')
            return glm_tool(prepared_data, family=family)
        elif regression_type.upper() == 'POISSON':
            return poisson_tool(prepared_data)
        elif regression_type.upper() == 'LOGIT':
            return logit_tool(prepared_data)
        elif regression_type.upper() == 'RIDGE':
            alpha = kwargs.get('alpha', 1.0)
            return ridge_tool(prepared_data, alpha=alpha)
        elif regression_type.upper() == 'LASSO':
            alpha = kwargs.get('alpha', 1.0)
            return lasso_tool(prepared_data, alpha=alpha)
        else:
            return f"지원되지 않는 회귀 유형: {regression_type}"
    except Exception as e:
        return f"회귀 분석 중 오류 발생: {str(e)}"
