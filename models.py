#models
from langchain_ollama import OllamaLLM
#select model
model = "gpt-oss:20b"

#user prompt
prompt = "Bryan Mbeumo brentford or Cunha wolves? Who is the better footballer and why?"



def run_model(model, prompt):


    try:
        # Initialise the model
        llm = OllamaLLM(model=model)


        # Run the model
        print(f"Asking model: '{prompt}'")
        print("---------------------------------")

        response = llm.invoke(prompt)

        #Print the response
        print(response)

    except Exception as e:
        print(f"An error occurred: {e}")
        print("\n=== Troubleshooting Tips ===")
        print("1. Did you forget to start the Ollama application?")
        print("2. Is the model name 'gpt-oss:20b' spelled correctly?")

    
run_model(model, prompt)


