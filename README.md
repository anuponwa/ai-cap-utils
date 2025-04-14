# do-cap-utils

A library to simplify your use case with chat bot with LLMs. Perfect for Common AI Platform.

## Installation
 ```
 uv add do-cap-utils
 ```

## Usage

See `example.py` and `example_multiagents.py`.


### General

```python
from do_cap_utils.agent.base import BaseAgent
from do_cap_utils.agent.prebuilt import SummariserAgent
from do_cap_utils.tool import tool


# --------------------------------------------
# Create and run your own agent with some tool
# --------------------------------------------
@tool
def add_numbers(a: int, b: int):
    """Adds two numbers together"""
    return a + b


agent = BaseAgent(
    system_prompt="You are a helpful assistant that answers everything nicely (only that you can do or know)",
    tools=[add_numbers],
)
r = agent.chat_each_message("hello")
r = agent.chat_each_message("add two numbers for me")
r = agent.chat_each_message("2 and 4")
print(r)  # Inpect the output
print()

# You can use `chat_interact()` method to have an interactive session
agent.chat_interact()
```

Now, if you use `self.chat_interact()`, save the file and run this file in the command line.

```
uv run your_file.py
```

Or activate your python environment and run
```
python your_file.py
```

Example output:
```
================================ Human Message =================================

hello
================================== Ai Message ==================================

Hello! How can I assist you today?
================================ Human Message =================================

add two numbers for me
================================== Ai Message ==================================

Of course! Please provide the two numbers you'd like to add.
================================ Human Message =================================

2 and 4
================================== Ai Message ==================================
Tool Calls:
  add_numbers (call_kGDJ89gTwG6cTZFT7cWr1dBs)
 Call ID: call_kGDJ89gTwG6cTZFT7cWr1dBs
  Args:
    a: 2
    b: 4
================================= Tool Message =================================
Name: add_numbers

6
================================== Ai Message ==================================

The sum of 2 and 4 is 6.
```