CG_start = """
I will give you two examples below. Please answer my questions according to the format of this example.
Example one:
    {"description": "Please return the sum of the two given numbers.",
    "imcomplete_code": "
    def sum(x, y):
        # TODO:Implement the addition operation of x and y and return the result.
        ",
    "complete_code": "
    def sum(x, y):
        return x + y
    "}
Example two:
    {"description": "Please return the product of two given numbers.",
    "imcomplete_code": "
    def multiply(x, y):
        # TODO:Implement the multiplication operation of x and y and return the result.
        ",
    "complete_code": "
    def multiply(x, y):
        return x * y
    "}
The complete_code part is the complete executable function code obtained after code completion of the incomplete_code part based on the description part.
Below is my description and incomplete_code part:\n
"""

CG_end = """\nPlease complete the unfinished code in the incomplete_code part according to the description in the description part, and write the final complete code result into complete_code.
The format of your output must be consistent with the example I gave, and do not output any additional information.Please output the results in json format.
I am writing the output format again:{"description": "","imcomplete_code": "","complete_code": ""}
"""

CM_start = """
I will give you two examples below. Please answer my questions according to the format of this example.
Example one:
    {"description": "Please return the sum of the two given numbers.",
    "bug_code": "
    def sum(x, y):
        return x * y + 3
    ",
    "corrected_code": "
    def sum(x, y):
        return x + y
    "}
Example two:
    {"description": "Please return the product of two given numbers.",
    "bug_code": "
    def multiply(x, y):
        return 2 * x + 3 * y
    ",
    "corrected_code": "
    def multiply(x, y):
        return x * y
    "}
Among them, the corrected_code part is the correct code obtained by correcting the problematic bug_code part according to the description part.
"""

CM_end = """\nPlease correct the problematic function in the bug_code part according to the function described in the description part, and write the finally corrected and executable code into the corrected_code part.
The format of your output must be consistent with the example I gave, and do not output any additional information. Please output the results in json format.
I am writing the output format again:{"description": "", "bug_code": "", "corrected_code": ""}
"""

CR_start = """
I will give you two examples below. Please answer my questions according to the format of this example.
Example one:
    {"requirement": "Check Python code snippets for syntax errors, logic errors, performance issues, and security vulnerabilities. Function description of the function: Please return the sum of the two given numbers.",
    "problem_code": "
    def sum(x, y):
        a(x)
        b(y)
        for i in range(10000):
            print(i)
        return x + y
    ",
    "performance_issues": "
    for i in range(10000):
        print(i)
    ",
    "security_issues": "
    a(x)
    b(y)
    ",
    "irregular_naming": "None",
    "logical_errors": "None"
    }
Example two:
    {"requirement": "Check Python code snippets for syntax errors, logic errors, performance issues, and security vulnerabilities. Function description of the function: Please return the product of two given numbers.",
    "problem_code": "
    def multiply(x, y):
        result_sum = x * y
        return 2 * x + 3 * y
    ",
    "performance_issues": "None",
    "security_issues": "None",
    "irregular_naming": "
    result_sum = x * y
    ",
    "logical_errors": "
    return 2 * x + 3 * y
    "
    }
Among them, the four fields of performance_issues, security_issues, irregular_naming and logical_errors are filled in according to the requirements part to review whether the code in the problem_code part has problems in these four aspects. If there is a problem in a certain aspect, write the specific problem code snippet into the corresponding area. If there is no problem in a certain aspect, fill in None in this area to indicate that there is no problem.
"""

CR_end = """\nPlease review the problematic function in the problem_code part according to the requirement part.
Errors are classified into performance_issues, security_issues, irregular_naming and logical_errors. If there is a problem in a certain aspect, point out the specific problematic code fragment and write it into the corresponding field; if there is no problem in this aspect, then write it into the corresponding field. Just fill in None.
The format of your output must be consistent with the example I gave, and do not output any additional information. Please output the results in json format.
I am writing the output format again:{"requirement": "", "problem_code": "", "performance_issues": "", "security_issues": "", "irregular_naming": "", "logical_errors": ""}
"""

CUF_start = """
I will give you two examples below. Please answer my questions according to the format of this example.
Example one:
{"code": "def sum(x, y):\n    return x + y\n", "input": "x=1, y=2", "output": "3"}
Example two:
{"code": "def multiply(x, y):\n    return x * y\n", "input": "x=1, y=2", "output": "2"}
Among them, the output part is the output result obtained after the execution of the function input from the input part to the code part. The following is my code and input modules:
"""

CUF_end = """\nPlease provide the final output part based on the input and code part. The format of your output must be consistent with the example I gave, and do not output any additional information.
I am writing the output format again:{"code": "", "input": "", "output": ""}
"""

CUR_start = """
I will give you two examples below. Please answer my questions according to the format of this example.
Example one:
{"code": "def sum(x, y):\n    return x + y\n", "input": "x=1, y=2", "output": "3"}
Example two:
{"code": "def multiply(x, y):\n    return x * y\n", "input": "x=1, y=2", "output": "2"}
Among them, the output part is the output result obtained after the execution of the function input from the input part to the code part. The following is my code and input modules:
"""

CUR_end = """\nPlease provide the final input part based on the output part and code part. The format of your output must be consistent with the example I gave, and do not output any additional information.
I am writing the output format again:{"code": "", "input": "", "output": ""}
"""

PROMPTS = {
    "CG_prompt": [CG_start, CG_end],
    "CM_prompt": [CM_start, CM_end],
    "CR_prompt": [CR_start, CR_end],
    "CUF_prompt": [CUF_start, CUF_end],
    "CUR_prompt": [CUR_start, CUR_end],
}