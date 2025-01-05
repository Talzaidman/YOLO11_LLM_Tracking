import openai
import json
from openai import OpenAI


def get_gpt_command(inp):
    # Set your API key
    key = ""
    client = OpenAI(
        api_key=key,  # This is the default and can be omitted
    )
    # Set your API key (make sure to keep it secure)
    openai.api_key = key

    content_gpt = r"Given the input, please return a structured output in JSON format. Ensure that the JSON output includes only the key-value pairs related to the main object of interest. example 1, input: \"i want to see the person\", answer: \"{\"class\": \"person\"}\". example 2, input: \"i'm interested in dogs\", answer: \"{\"class\": \"dog\"}\"  input: "
    content_gpt += inp

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": content_gpt,
            }
        ],
        model="gpt-4o-mini",
    )

    # Print the response
    #print(chat_completion.choices[0].message.content)
    response_content = chat_completion.choices[0].message.content
    cleaned_string = response_content.strip('```json\n').strip('```').strip()

    # Parse the JSON response
    json_response = json.loads(cleaned_string)

    # Print the parsed JSON
    print(json_response)
    return json_response