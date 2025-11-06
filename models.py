# models
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
import player_select as ps

# select model
model = "gpt-oss:20b"





mid_data = ps.mid_data.head(25)
fwd_data = ps.fwd_data.head(25)
gkp_data = ps.gkp_data.head(25)
def_data=ps.def_data.head(25)

# system prompt for higher reasoning (adjust as needed; options: low, medium, high)
system_prompt = "Reasoning: high"

# user prompt (midfield selection)
mid_prompt = f"""You are an FPL (football/soccer game) expert. Your task is to analyse the following table of player data that shows fixture adjusted performance over the next X games and select the best 5 midfieders (players with only position MID).

Rules: 
- Consider 'creativity+threat' as a primary decision factor.
- Consider 'creativity+threat/price' as another primary decision factor.


Here is the attack player data table:

{mid_data.to_string()}

give your answer only as the list of web_names of the top 5 midfielders, no other text.

format guidelines:
web_name1, web_name2, web_name3, web_name4, web_name5

"""


# user prompt (attacker selection)
fwd_prompt = f"""You are an FPL (football/soccer game) expert. Your task is to analyse the following table of player data that shows fixture adjusted performance over the next X games and select the best 3 attackers (players with only position FWD).

Rules: 
- Consider 'creativity+threat' as a primary decision factor.
- Consider 'creativity+threat/price' as another primary decision factor.


Here is the attack player data table:

{fwd_data.to_string()}

give your answer only as the list of web_names of the top 3 attackers, no other text.

format guidelines:
web_name1, web_name2, web_name3

"""

# user prompt (defesnse selection)
def_prompt = f"""You are an FPL (football/soccer game) expert. Your task is to analyse the following table of player data that shows fixture adjusted performance over the next X games and select the best 5 defenders (players with only position DEF).

Rules: 
- Consider 'defense_adjusted_points' as a primary decision factor.
- Consider 'defense_adj_pts_per_million' as another primary decision factor.


Here is the defense player data table:

{def_data.to_string()}

give your answer only as the list of web_names of the top 5 defenders, no other text.

format guidelines:
web_name1, web_name2, web_name3, web_name4, web_name5

"""

# user prompt (GK selection)
gkp_prompt = f"""You are an FPL (football/soccer game) expert. Your task is to analyse the following table of player data that shows fixture adjusted performance over the next X games and select the best 2 goalkeepers (players with only position GKP).

Rules: 
- Consider 'defense_adjusted_points' as a primary decision factor.
- Consider 'defense_adj_pts_per_million' as another primary decision factor.


Here is the defense player data table:

{gkp_data.to_string()}

give your answer only as the list of web_names of the top 2 goalkeepers, no other text.

format guidelines:
web_name1, web_name2

"""








def run_model(model, system_prompt, user_prompt):
    try:
        # Initialise the model (add temperature=0 for more deterministic outputs if needed)
        llm = ChatOllama(model=model, temperature=0.2)

        # Combine into messages
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

        # Run the model
        #print(f"Asking model with messages: {messages}")
        print("---------------------------------")

        response = llm.invoke(messages)

        # Print the response
        print(response.content)

    except Exception as e:
        print(f"An error occurred: {e}")
    
    return response.content


gkp_string = run_model(model, system_prompt, gkp_prompt)

def_string = run_model(model, system_prompt, def_prompt)

mid_string = run_model(model, system_prompt, mid_prompt)

fwd_string = run_model(model, system_prompt, fwd_prompt)
