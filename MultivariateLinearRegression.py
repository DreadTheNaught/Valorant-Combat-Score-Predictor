# %% [markdown]
# # Multivariate Linear Regression

# %% [markdown]
# ### Data to be used
# 
# Independent variable : Combat score
# 
# Dependent variables: kills, damage, headshot %, loadout value

# %%
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import json

# %%
# Open the JSON file containing the data
file = open("data.json")

# Load the data from the JSON file
data = json.load(file)

# Access the 'matches' data from the loaded JSON
matches = data['data']

# Define the player's unique identifier (PUUID)
puuid = '6ffcd21d-1888-516a-b8ac-c49fd1a7220d'

# Initialize empty lists to store round stats and combat scores
round_stats = []
combat_scores = []

# Iterate through each match in the data
for match in matches:
    # Check if the match is not a Deathmatch or Team Deathmatch mode
    if match["metadata"]["mode"] != "Deathmatch" and match["metadata"]["mode"] != "Team Deathmatch":
        # Access the rounds within the match
        rounds = match['rounds']

        # Iterate through each round within the match
        for round in rounds:
            # Access the player stats for each round
            round_player_stats = round['player_stats']

            # Iterate through each player's stats in the round
            for player in round_player_stats:
                # Check if the player's PUUID matches the desired PUUID
                if player['player_puuid'] == puuid:
                    # Extract relevant statistics for the player
                    kills = player['kills']
                    damage = player['damage']

                    # Calculate the headshot percentage
                    if player['headshots'] + player['bodyshots'] + player['legshots'] == 0:
                        headshot_percent = 0
                    else:
                        headshot_percent = player['headshots'] / (
                            player['headshots'] + player['bodyshots'] + player['legshots'])

                    loadout_value = player['economy']['loadout_value']

                    # Store the player's statistics for the round
                    player_values = [kills, damage,
                                     headshot_percent, loadout_value]
                    round_stats.append(player_values)

                    # Store the player's combat score
                    combat_scores.append(player['score'])

# %%
#Assigning the dependent variables (X) and independent variable (y)
X = round_stats
y = combat_scores

# %% [markdown]
# ## Multivariate Linear Regression

# %%
# Enter Variable to be displayed
variables = ("Kills", "Damage", "Headshot %", "Loadout Value")

variable_number = 3

# %%
#Creating the model
pipe = Pipeline([
    ("scale", StandardScaler()),
    ("model", LinearRegression())
])


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state= 45)

#Training
pipe.fit(X_train, y_train)

#Predicting
y_pred = pipe.predict(X_test)

# %%
#Plotting the Multivariate regression

plt.scatter([item[variable_number]
            for item in X_train], y_train, label='Training Data')
plt.scatter([item[variable_number] for item in X_test],
            y_pred, color='red', label='Predicted Data')
plt.scatter([item[variable_number] for item in X_test],
            y_test, color='green', label='Actual Data')

plt.title("Valorant Data")
plt.xlabel(variables[variable_number])
plt.ylabel("Combat Score")
plt.legend()

plt.show()
