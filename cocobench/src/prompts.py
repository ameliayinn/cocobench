def CG_prompt(prompt: str) -> str:
    return f"""Complete the following Python code.\n{prompt}"""

def CM_prompt(prompt: str) -> str:
    return f"""Modify the following Python code.\n{prompt}"""

def CR_prompt(prompt: str) -> str:
    return f"""Review the following Python code.\n{prompt}"""

def CU_prompt(prompt: str) -> str:
    return f"""Explain the following Python code.\n{prompt}"""