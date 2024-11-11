### Experiment Diverse Search
# The poor performance of the evolved agents are assumed to be due to the evolution only being 
# performed through mutation (no crossover), which just does a fuzzy search and is likely to get stuck in local optima. 

# Test solutions:
# 1. Try adding cross over
# 2. Use something like CMA-ES to use samples from a larger space.

# Crossover is usually done from better performing agents, but none of the agents perform well,
# so parents are just chosen at random.

import os
import nmmo
from nmmo.render.replay_helper import FileReplayHelper
import numpy as np
import random
import copy
import pickle
import pandas as pd

from real_time_evolution import mutate, pick_best, simple_mutate, crossover, crossoverModelDict
from agent_neural_net import get_input, PolicyNet, save_state, NoCombatNet
from logging_functions import EraLogger, calculate_avg_lifetime, GetAgentXP
from config import set_config

replay_helper = FileReplayHelper()
import torch
torch.set_num_threads(1)

config, NPCs = set_config()

MATURE_AGE = 10000
INTERVAL = 30
EXP_NAME = 'DiverseSearch'

env = nmmo.Env()
player_N = env.config.PLAYER_N

obs = env.reset()#[0]

env.realm.record_replay(replay_helper)

# Define the model
output_size = 5
output_size_attack = player_N+1+NPCs

# Random weights with a FF network
model_dict = {i+1: PolicyNet(output_size, output_size_attack)  for i in range(player_N)} # Dictionary of random models for each agent

# Forward pass with a feed forward NN
action_list = []
action_list_attack = []

for i in range(env.config.PLAYER_N):
    if (env.realm.players.corporeal[1].alive):
        # Get the observations
        inp = get_input(env.realm.players.entities[i+1], obs[i+1]['Tile'], obs[i+1]['Entity'],env.realm.players.entities[i+1].pos)
        # Get move actions
        output, style, output_attack = model_dict[i+1](inp)
        # Get attack actions (target, since agents only do melee combat)
        action_list.append(output)
        action_list_attack.append(output_attack)

actions = {}
for i in range(env.config.PLAYER_N):
    actions[i+1] = {"Move":{"Direction":1}, "Attack":{"Style":0,"Target":int(action_list_attack[i])}}

replay_helper.reset()

life_durations = {i+1: 0 for i in range(env.config.PLAYER_N)}
birth_interval = {i+1: 0 for i in range(env.config.PLAYER_N)}

# Getting spawn positions
spawn_positions = [(0,0)]
for i in range(player_N):
    spawn_positions.append(env.realm.players.entities[i+1].spawn_pos)


# Setting up the average lifetime dictionary
avg_lifetime = {}

# Set up a list of all visited tiles by all agents
all_visited = []

# Set up max lifetime
max_lifetime = 0
max_xp = 0
oldest = []
pop_exp = []
pop_life = []
max_lifetime_dict = {}

steps = 50_001 #steps = 10_000_001


AVAILABLE = False

### Era logging
eraLogger = EraLogger(startStep = 0)
unsavedEraData = []
unsavedAgentEraData = [] #Updated when agent dies or at end of era
def GetAgentData(entity, id):
    agentData = GetAgentXP(entity)
    agentData.update({
        'era': eraLogger.curretEra,
        'age': life_durations[id],
    })
    return agentData

def UpdateEraData():
    eraData = eraLogger.GetCurrentEraData(
        total_steps=step, 
        living_agents=env.num_agents)
    unsavedEraData.append(eraData)

    for i in range(player_N):
        if i+1 in env.realm.players.entities:
            unsavedAgentEraData.append(
                GetAgentData(env.realm.players.entities[i+1], i+1)
            )
firstSave = True
###

#Save data to output directory
output_dir = os.path.join('output', EXP_NAME)
os.makedirs(output_dir, exist_ok=True)

avail_index = []
# The main loop
for step in range(steps):
    if env.num_agents ==0 :
        eraLogger.exctinct = True
        UpdateEraData()

        print('extinction')

        model_dict = crossoverModelDict(model_dict, player_N)
        for i in range(player_N):
            simple_mutate(i+1, model_dict, alpha=0.01)
        env.close()
        env = nmmo.Env()
        obs = env.reset()#[0]
    else: 
        eraLogger.exctinct =  False # Set extinct to false after one iteration, to prevent loggin exctinct eras twice


    #If the number of agents alive doesn't correspond to PLAYER_N, spawn new offspring
    if env.num_agents != player_N:
        AVAILABLE = True
        avail_index = []
        for i in range(player_N):
            ## ignore actions of unalive agents
            if i+1 not in env.realm.players.entities:
                avail_index.append(i+1)
    else:
        AVAILABLE = False

    XP_SUM = 0
    LIFE_SUM = 0 
    
    if step%100==0:
        print(step) 
        with open(os.path.join(output_dir, EXP_NAME + 'timestep.txt'), 'w') as file:
            file.write(str(step))


    current_oldest = life_durations[max(life_durations, key=life_durations.get)]

    #Save era data to file
    if (step+1)%10_000 == 0:
        # This could be called on exit instead (with try: except: keyboard interrupt), but might be undesirable on the cluster.
        eraFilePath = os.path.join(output_dir, EXP_NAME + 'era_data.csv')
        agentFilePath = os.path.join(output_dir, EXP_NAME + 'agent_data.csv')
        print(unsavedAgentEraData)

        if firstSave:
            df = pd.DataFrame(unsavedEraData)
            df.to_csv(eraFilePath) 
            df = pd.DataFrame(unsavedAgentEraData)
            df.to_csv(agentFilePath) 
            firstSave = False
        else:
            df = pd.DataFrame(unsavedEraData)
            df.to_csv(eraFilePath, mode='a', header=False) #Not sure this index will work
            df = pd.DataFrame(unsavedAgentEraData)
            df.to_csv(agentFilePath, mode='a', header=False) 
        unsavedEraData = []
        unsavedAgentEraData = []


    if (step + 1- eraLogger.eraStartStep) % 500 == 0: #if (step + 1- eraLogger.eraStartStep) % 10_000 == 0:
        UpdateEraData()

        print('reset env') 
        env.close()
        env = nmmo.Env()
        obs = env.reset()#[0]
        

    for i in range(player_N):
        ## ignore actions of unalive agents
        if i+1 not in env.realm.players.entities:
            life_durations[i+1] = 0
            birth_interval[i+1] = 0
            actions[i+1] = {}

        # Check if agents are alive, and if someone dies ignore their action
        elif i+1 in env.realm.players.entities and i+1 in obs:
            life_durations[i+1] += 1
            birth_interval[i+1] += 1

            XP_SUM += env.realm.players.entities[i+1].melee_exp.val
            XP_SUM += env.realm.players.entities[i+1].range_exp.val
            XP_SUM += env.realm.players.entities[i+1].mage_exp.val
            XP_SUM += env.realm.players.entities[i+1].fishing_exp.val
            XP_SUM += env.realm.players.entities[i+1].herbalism_exp.val
            XP_SUM += env.realm.players.entities[i+1].prospecting_exp.val
            XP_SUM += env.realm.players.entities[i+1].carving_exp.val
            XP_SUM += env.realm.players.entities[i+1].alchemy_exp.val

            LIFE_SUM += env.realm.players.entities[i+1].time_alive.val 
            if env.realm.players.entities[i+1].time_alive.val > max_lifetime:
                max_lifetime = env.realm.players.entities[i+1].time_alive.val
 
            ##if conditions are right make an offspring
            if len(avail_index)>0 and life_durations[i+1] > MATURE_AGE and birth_interval[i+1] > INTERVAL:
                birth_interval[i+1] =0
                new_born = avail_index[0] 
                avail_index.pop(0)
                parent = i+1

                #unsavedAgentEraData.append(GetAgentData(new_born)) #Store data of dead agent before rebirth

                try:
                    # Spawn individual in the same place as parent
                    x, y = env.realm.players.entities[parent].pos
                    #x, y = random.choice(spawn_positions)
                except:
                    # Spawn individual at a random spawn location
                    x, y = random.choice(spawn_positions)
                    spawn_positions[new_born] = (x,y)

                env.realm.players.spawn_individual(x, y, new_born)

                life_durations[new_born] = 0
                birth_interval[new_born] = 0
                model_dict[new_born] = crossover(model_dict[new_born], model_dict[parent])
                model_dict[new_born].hidden = (torch.zeros(model_dict[new_born].hidden[0].shape), torch.zeros(model_dict[new_born].hidden[1].shape))
                simple_mutate(new_born, model_dict, alpha=0.1)
                eraLogger.birthTracker += 1



            inp = get_input(env.realm.players.entities[i+1], obs[i+1]['Tile'], obs[i+1]['Entity'], env.realm.players.entities[i+1].pos)
            output, style, output_attack = model_dict[i+1](inp)

            ### action_list.append(output)
            actions[i+1] = {"Move":{"Direction":int(output)} , "Attack":{"Style":style,"Target":int(output_attack)}, "Use":{"InventoryItem":0}}

        else: 
            actions[i+1] = {}

    # Run a step
    obs, rewards, dones, infos = env.step(actions) ##TODO: why not use DONES to replace?

    ###! Does not work, indexing into dead agents cause errors
    # for id, done in dones.items():
    #     if done:
    #         unsavedAgentEraData.append(
    #             GetAgentData(env.realm.players.entities[id], id)
    #         )

    pop_exp.append(XP_SUM)
    pop_life.append(LIFE_SUM)
    oldest.append(max_lifetime)


    if (step+1)%3000==0:
      pickle.dump((pop_exp, pop_life, oldest), open(os.path.join(output_dir, 'progress.pkl'),'wb'))
      print('save replay')
      if step < 5000:
        replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step)), compress=False)
      else:
        replay_helper.save(os.path.join(output_dir, EXP_NAME), compress=False)

      replay_helper = FileReplayHelper()
      env.realm.record_replay(replay_helper)
      replay_helper.reset()
    if (step+1)%100_000==0:
      print('save population weights')
      pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_'+str(step)+'.pickle'),'wb'))


# Save replay file and the weights

#replay_file = f"/content/replay1"
#replay_helper.save("no_brain22", compress=False)
#save_state(model_dict, f"weights")
pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_final.pickle'),'wb'))