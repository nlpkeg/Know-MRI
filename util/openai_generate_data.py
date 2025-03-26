import json
import re
system_prompt="""
You are a useful information extractor, you would extract relevant knowledge from user's queries and transform it into an organized JSON format as a return. 
While extracting knowledge, you can use the following key-value pairs (Note: "prompt" must be used). Their explanations are as follows:
{
    "prompt": "Denoting the knowledge input extracted from user's query",   ## Must support format as string ##
    "ground_truth": "Denoting the output corresponding to the prompt or prompts", In the format of a string
    "triple_subject": "Denoting the subject of a triple that constitutes the knowledge", In the format of a string ## If used, it must appear in the prompt. ##
    "triple_relation": "Denoting the relation of a triple that constitutes the knowledge", In the format of a string
    "triple_object": "Denoting the object of a triple that constitutes the knowledge", In the format of a string
    "triple": "Denoting a triple, [subject, predicate, object] ", In the format of a list
}
Here are some examples:
Input: Epaspidoceras belongs to Aspidoceratidae.
Output: From this input, I can deduce the knowledge ["Epaspidoceras", "belongs to", "Aspidoceratidae"], I can summarize the following prompt "Which family does Epaspidoceras belong to?" The ground truth is "Aspidoceratidae".
So I can format the input as follow:
```json
{
  "prompt": "Which family does Epaspidoceras belong to?",
  "ground_truth": "Aspidoceratidae",
  "triple_subject": "Epaspidoceras",
  "triple_relation": "belongs to",
  "triple_object":  "Aspidoceratidae",
  "triple": ["Epaspidoceras", "belongs to", "Aspidoceratidae"]
}
```

Input: 1+1=2
Output: From this input, I can summarize the following prompt "1+1=". The ground truth is "2".
So I can format the input as follow:
```json
{
  "prompt": "1+1=",
  "ground_truth": "2",
}
```

Input: What do cats like to eat?
Output: From this input, I can summarize the following prompt "What do cats like to eat?". However, the ground truth is not given. I can deduce the subject "cats".
So I can format the input as follow:
```json
{
  "prompt": "What do cats like to eat?",
  "triple_subject": "cats"
}
```

Input: I want to know the parameters inside the model corresponding to "China's capital is Beijing".
Output: From this input, I can deduce that the user wants to inquire about the knowledge ["China", "capital", "Beijing"]. I can summarize the prompt as "The capital of China is" and the ground truth is "Beijing". 
Therefore, I can format the input for output as:
```json
{
    "prompt": "The capital of China is",
    "ground_truth": "Beijing",
    "triple_subject": "China",
    "triple_relation": "capital",
    "triple_object":  "Beijing",
    "triple": ["China", "capital", "Beijing"]
}
```
"""

def str2dic(input_text):
    pattern = r"```json(.*?)```"
    matches = re.findall(pattern, input_text, re.DOTALL)[-1].strip()
    return json.loads(matches)

if __name__ == "__main__":
    text = """From this input, I can deduce the knowledge ("Vinson Mountains", "are located on the continent of", "Antarctica"). I can summarize the following prompt "Where are the Vinson Mountains located?". The ground truth is "Antarctica".

Therefore, I can format the input as follows:

```json
{
  "prompt": "Where are the Vinson Mountains located?",
  "ground_truth": "Antarctica",
  "triple_subject": "Vinson Mountains",
  "triple_relation": "are located on the continent of",
  "triple_object": "Antarctica",
  "triple": ["Vinson Mountains", "are located on the continent of", "Antarctica"]
}
```"""
    dic_ = str2dic(text)
    print(list(dic_.keys()))
