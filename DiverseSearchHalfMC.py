### Experiment Diverse Search
# The poor performance of the evolved agents are assumed to be due to the evolution only being 
# performed through mutation (no crossover), which just does a fuzzy search and is likely to get stuck in local optima. 

### Experiment DiverseSearchHalfMC
# Current version of DivSearch had let agents that reached a treshold reappear in the next era. 
# However these reused agents were likely to not do well at their new spawn location, and eventually their succesful genes would die out.
# This half MC version, allows agents that reach half the treshold to be reused, unless the treshold is reached, where at the threshold will doube again.

# This version does not have births

import os
import time
import nmmo
from nmmo.render.replay_helper import FileReplayHelper
import numpy as np
import random
import copy
import pickle
import pandas as pd
import gc

from real_time_evolution import mutate, pick_best, simple_mutate, crossover, crossoverModelDict
from agent_neural_net import get_input, PolicyNet, save_state, NoCombatNet
from logging_functions import EraLogger, SaveData, GetAgentXP
from config import set_config

import torch
torch.set_num_threads(1)

config, NPCs = set_config()

MATURE_AGE = 10000000
INTERVAL = 30
EXP_NAME = 'DiverseSearchHalfMC'

env = nmmo.Env()
player_N = env.config.PLAYER_N

### Environment reset
def reset_env(env):
    env.close()
    env = nmmo.Env()
    obs = env.reset()#[0]
    replay_helper = FileReplayHelper()
    env.realm.record_replay(replay_helper)
    replay_helper.reset()
    return env, replay_helper, obs
###

env, replay_helper, obs = reset_env(env)

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

life_durations = {i+1: 0 for i in range(env.config.PLAYER_N)}
birth_interval = {i+1: 0 for i in range(env.config.PLAYER_N)}

# Getting spawn positions
spawn_positions = [(0,0)]
for i in range(player_N):
    spawn_positions.append(env.realm.players.entities[i+1].spawn_pos)

steps = 1_000_001
agentSuccess = 100

### Era logging
eraLogger = EraLogger()
unsavedEraData = []
#unsavedAgentEraData = [] #Updated when agent dies or at end of era
def GetAgentData(entity, id):
    agentData = GetAgentXP(entity)
    agentData.update({
        'era': eraLogger.curretEra,
        'age': life_durations[id],
    })
    return agentData

def UpdateEraData(step, env):
    eraData = eraLogger.GetCurrentEraData(
        total_steps=step, 
        living_agents=env.num_agents)
    unsavedEraData.append(eraData)

    # for i in range(player_N):
    #     if i+1 in env.realm.players.entities:
    #         unsavedAgentEraData.append(
    #             GetAgentData(env.realm.players.entities[i+1], i+1)
    #         )
firstSave = True
###

#Save data to output directory
output_dir = os.path.join('output', EXP_NAME)
os.makedirs(output_dir, exist_ok=True)

avail_index = [] #This is used both for dead and living agents. Can be unclear
AVAILABLE = False
# The main loop
for step in range(steps):
    if env.num_agents ==0 :
        UpdateEraData(step, env)

        if AVAILABLE:
            for i in range(player_N):
                if i+1 not in avail_index:
                    if len(avail_index) < 2: 
                        model_dict[i + 1] = model_dict[avail_index[0]]
                        model_dict[i + 1].hidden = (torch.zeros(model_dict[id].hidden[0].shape), torch.zeros(model_dict[id].hidden[1].shape))
                    else:
                        parent1_id, parent2_id = random.sample(avail_index, 2)
                        model_dict[i + 1] = crossover(model_dict[parent1_id], model_dict[parent2_id])

                        model_dict[i + 1].hidden = (torch.zeros(model_dict[id].hidden[0].shape), torch.zeros(model_dict[id].hidden[1].shape))
        else:
            model_dict = crossoverModelDict(model_dict, player_N)
            for i in range(player_N):
                simple_mutate(i+1, model_dict, alpha=0.01)
        env, replay_helper, obs = reset_env(env)
        avail_index = []
        AVAILABLE = False
    
    if step%300==0:
        print(f'step {step}, maxAge {agentSuccess}, era starting step {eraLogger.eraStartStep}') 

    #Save era data to file
    if (step+1)%10_000 == 0:
        unsavedEraData, firstSave = SaveData(output_dir, EXP_NAME, unsavedEraData, firstSave)

    if (step + 1 - eraLogger.eraStartStep) % agentSuccess > 0.5 and not AVAILABLE:
        AVAILABLE = True
        avail_index = list(env.realm.players.entities.keys())


    if (step + 1 - eraLogger.eraStartStep) % agentSuccess == 0: #if (step + 1- eraLogger.eraStartStep) % 10_000 == 0:
        print(f'\n{agentSuccess} successful steps') 
        print(f'\nStep: {step}, start step: {eraLogger.eraStartStep}, diff: {step - eraLogger.eraStartStep}') 

        UpdateEraData(step, env)

        avail_index = []
        for i in range(player_N):
            ## ignore actions of unalive agents
            if i+1 not in env.realm.players.entities:
                avail_index.append(i+1)
        #REPLACE DEAD AGENTS
        for id in avail_index:
            if len(env.realm.players.entities.keys()) == 1: 
                model_dict[id] = model_dict[list(env.realm.players.entities.keys())[0]]
                model_dict[id].hidden = (torch.zeros(model_dict[id].hidden[0].shape), torch.zeros(model_dict[id].hidden[1].shape))
            elif len(env.realm.players.entities.keys()) < 2:
                break # This should never trigger, but it does :(
            else:
                parent1_id, parent2_id = random.sample(env.realm.players.entities.keys(), 2)
                model_dict[id] = crossover(model_dict[parent1_id], model_dict[parent2_id])
                model_dict[id].hidden = (torch.zeros(model_dict[id].hidden[0].shape), torch.zeros(model_dict[id].hidden[1].shape))
        
        replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step) + "_" + str(agentSuccess)), compress=False)
        env, replay_helper, obs = reset_env(env)
        avail_index = []
        AVAILABLE = False
        agentSuccess *= 2
        
    ### Respawns / Births
    # for i in range(player_N):
    #     ## ignore actions of unalive agents
    #     if i+1 not in env.realm.players.entities:
    #         life_durations[i+1] = 0
    #         birth_interval[i+1] = 0
    #         actions[i+1] = {}

    #     # Check if agents are alive, and if someone dies ignore their action
    #     elif i+1 in env.realm.players.entities and i+1 in obs:
    #         life_durations[i+1] += 1
    #         birth_interval[i+1] += 1
 
    #         ##if conditions are right make an offspring
    #         if len(avail_index)>0 and life_durations[i+1] > MATURE_AGE and birth_interval[i+1] > INTERVAL:
    #             birth_interval[i+1] =0
    #             new_born = avail_index[0] 
    #             avail_index.pop(0)
    #             parent = i+1

    #             #unsavedAgentEraData.append(GetAgentData(new_born)) #Store data of dead agent before rebirth

    #             try:
    #                 # Spawn individual in the same place as parent
    #                 x, y = env.realm.players.entities[parent].pos
    #                 #x, y = random.choice(spawn_positions)
    #             except:
    #                 # Spawn individual at a random spawn location
    #                 x, y = random.choice(spawn_positions)
    #                 spawn_positions[new_born] = (x,y)

    #             env.realm.players.spawn_individual(x, y, new_born)

    #             life_durations[new_born] = 0
    #             birth_interval[new_born] = 0
    #             model_dict[new_born] = crossover(model_dict[new_born], model_dict[parent])
    #             model_dict[new_born].hidden = (torch.zeros(model_dict[new_born].hidden[0].shape), torch.zeros(model_dict[new_born].hidden[1].shape))
    #             simple_mutate(new_born, model_dict, alpha=0.1)
    #             eraLogger.birthTracker += 1

    #         inp = get_input(env.realm.players.entities[i+1], obs[i+1]['Tile'], obs[i+1]['Entity'], env.realm.players.entities[i+1].pos)
    #         output, style, output_attack = model_dict[i+1](inp)

    #         ### action_list.append(output)
    #         actions[i+1] = {"Move":{"Direction":int(output)} , "Attack":{"Style":style,"Target":int(output_attack)}, "Use":{"InventoryItem":0}}
    #     else: 
    #         actions[i+1] = {}

    # Run a step
    obs, rewards, dones, infos = env.step(actions) ##TODO: why not use DONES to replace?

    ###! Does not work, indexing into dead agents cause errors
    # for id, done in dones.items():
    #     if done:
    #         unsavedAgentEraData.append(
    #             GetAgentData(env.realm.players.entities[id], id)
    #         )


replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step) + "_" + str(agentSuccess)), compress=False)
replay_helper = FileReplayHelper()
env.realm.record_replay(replay_helper)
replay_helper.reset()

pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_final.pickle'),'wb'))