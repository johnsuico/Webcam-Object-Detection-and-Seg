from llama_cpp import Llama

# Initialize the model
llm = Llama(model_path="Qwen3.5-9B-UD-Q4_K_XL.gguf", n_ctx=2048)

user_input = "Can you point to the red cup? Pretend there is an image of a red cup in front of you"

# --- Output 1: Natural Language ---
prompt_1 = f"User: {user_input}\nAssistant:"
response_1 = llm(
    prompt_1, 
    max_tokens=50, 
    stop=["User:"], # Stop generation if it tries to impersonate the user
    echo=False
)

text_output = response_1['choices'][0]['text'].strip()
print(f"Output 1: \"{text_output}\"")


# --- Output 2: Structured JSON ---
# We use a grammar here to FORCE the model to output valid JSON
# json_schema = """
#   root ::= "{" space "subject:" space string "," space "verb:" space string "}" space
#   string ::= "\"" ([^"]*) "\""
#   space ::= " "?
# """

# prompt_2 = f"Extract the subject and verb from this sentence: '{user_input}'. Return only JSON."

# response_2 = llm(
#     prompt_2, 
#     max_tokens=50, 
#     grammar=json_schema, # This ensures the output is strictly formatted
#     temperature=0.1      # Low temp for structure
# )

# json_output = response_2['choices'][0]['text'].strip()
# print(f"Output 2: {json_output}")