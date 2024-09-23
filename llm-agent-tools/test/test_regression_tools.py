# tools/regression_tool/create_regression_tools.py

from langchain.agents import Tool
from tools.regression_tool import (
    ols_tool,
    wls_tool,
    gls_tool,
    glm_tool,
    poisson_tool,
    logit_tool,
    ridge_tool,
    lasso_tool,
    common_interface
)

def create_regression_tools():
    # 종속 변수 선택 도구
    analyze_tool = Tool(
        name="Analyze Regression",
        func=common_interface.analyze_regression,
        description=(
            "Automatically select the dependent variable and perform the specified regression analysis. "
            "Provide 'data' as a dictionary and 'regression_type' (e.g., 'OLS', 'GLM'). "
            "Additional parameters can be provided as needed, such as 'weights' for WLS or 'family' for GLM."
        )
    )
    
    return [analyze_tool]
