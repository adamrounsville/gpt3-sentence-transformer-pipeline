import time
import openai
import pandas as pd
from foi_simple import *
from keys import *

openai.api_key = keys['openAI']

NUM_RESPONSES = 1 # Number of responses/prompt
PROMPT = ""

def do_query( PROMPT, max_tokens=512, engine="davinci" ):
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        temperature=0.7,
        max_tokens=max_tokens,
        top_p=1,
        logprobs=100,
    )
    return response

# ==============================================================================

responses = []

for i in range(NUM_RESPONSES): 
    try:
        response = do_query( PROMPT, max_tokens=128, engine="text-davinci-003" )
        resp_text = response.choices[0]['text']
        print(resp_text)
        responses.append(resp_text)
    except:
        time.sleep(5.0)

newdf = pd.DataFrame({'response': responses})
newdf.to_csv("./fake_data.csv")
