import requests
import json

# Define the URL and the headers
url = "http://localhost:8000/v1/chat/completions"
headers = {
    "Content-Type": "application/json"
}

tools = [{"arguments":{"discount_percentage":{"required":True,"type":"number","description":"The percentage discount to be applied"},
"original_price":{"description":"The original price of the item","required":True,"type":"number"}},
"description":"Calculate the discounted price of an item based on the original price and discount percentage","name":"calculate_discount"},
{"arguments":{"pod_name":{"description":"The name of the pod to be restarted","required":True,"type":"str"}},
"description":"A function to restart a given pod, useful for deployment and testing.","name":"restart_pod"}]

# Define the system prompt and the user message
system_prompt = f"""You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke:
{json.dumps(tools, indent=2)}"""

messages = [
    {
        "role": "system",
        "content": system_prompt
    },
    {
        "role": "user",
        "content": "What is the discounted price of the jacket, given it was originally $200 and there is a 20% reduction?"
    }
]

# Define the payload
data = {
    "model": "saudsaleem/bitagent-autotrain",
    "messages": messages
}
# print('data', data['messages'])
# Make the POST request
response = requests.post(url, headers=headers, data=json.dumps(data))
response = response.json()
# Print the response (or handle it as needed)
print(response['choices'][0]['message']['content'].strip())
