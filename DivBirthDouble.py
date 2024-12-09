from collections import deque
from datetime import datetime
import os
import pickle
import random
from logging_functions import EraLoggerV2
import nmmo
from nmmo.render.replay_helper import FileReplayHelper

from agent_neural_net import PolicyNet, get_input
from config import set_config, set_config_Big

import torch

from real_time_evolution import crossover, simple_mutate
torch.set_num_threads(1)


### Save destination.
EXP_NAME = 'DivBirthDouble'
startTime = datetime.now() 
output_dir = os.path.join(os.path.join('output', startTime.strftime("%m-%d")), EXP_NAME)
os.makedirs(output_dir, exist_ok=True)
MATURE_AGE = 50
INTERVAL = 30
###

config, NPCs = set_config_Big()
env = nmmo.Env() 
player_N = config.PLAYER_N
output_size = 5
output_size_attack = player_N+1+NPCs
# Random weights with a FF network
model_dict = {i+1: PolicyNet(output_size, output_size_attack)  for i in range(player_N)} # Dictionary of random models for each agent
# LOAD DICT FROM FILE TO CONTINUE TRAINING
#model_dict = pickle.load(open(EXP_NAME+'_agents_model_dict.pickle','rb'))

### Run one era
def RunEra(startStep, env, model_dict, eraLogger, longestEra):
    step = startStep
    obs = env.reset()
    deadAgents = deque([])
    agentAges = {i+1: 0 for i in range(env.config.PLAYER_N)}
    agentBirthInterval = {i+1: 0 for i in range(env.config.PLAYER_N)}

    replay_helper = FileReplayHelper()
    env.realm.record_replay(replay_helper)
    replay_helper.reset()

    while True:
        deadAgents.extend(env._dead_this_tick.keys())
        if env.num_agents == 0:
            print("Extinction")
            for agent in model_dict.keys():
                model_dict = simple_mutate(agent, model_dict, alpha=0.01)
                model_dict[agent].hidden = (torch.zeros(model_dict[agent].hidden[0].shape), torch.zeros(model_dict[agent].hidden[1].shape))
            eraLogger.UpdateEraData(step-startStep, len(env.realm.players.entities))

            if (step - startStep) > longestEra:
                replay_helper.save(os.path.join(output_dir, EXP_NAME + str(longestEra)), compress=False)
                longestEra *= 1.2
                print("Replay Saved")
            break

        if step%300==0:
            print(f'step {step}, era starting step {startStep}') 

        #Save era data to file
        if (step+1)%10_000 == 0:
            eraLogger.SaveToFiles(output_dir, EXP_NAME)

        #Save population weights
        if (step+1)%100_000==0:
            print('save population weights')
            pickle.dump(model_dict,open(EXP_NAME+'_agents_model_dict'+'.pickle','wb'))
        
        # Get actions from models
        actions = {}
        for agent in env.realm.players.entities.keys():
            inp = get_input(env.realm.players.entities[agent], obs[agent]['Tile'], obs[agent]['Entity'], env.realm.players.entities[agent].pos)
            output, style, output_attack = model_dict[agent](inp)
            actions[agent] = {"Move":{"Direction":int(output)} , "Attack":{"Style":style,"Target":int(output_attack)}, "Use":{"InventoryItem":0}}       

        # increment agent ages
        for agent in env.realm.players.entities.keys():
            agentAges[agent] += 1
            agentBirthInterval[agent] += 1

        # Respawn dead agents, if applicable. NOTE, respawned agents should not have actions
        while len(deadAgents) > 0:
            matureAgents = {key: value for key, value in agentAges.items() if value > MATURE_AGE}
            birthReadyAgents = {key: value for key, value in matureAgents.items() if agentBirthInterval[key] > INTERVAL}

            if len(birthReadyAgents) != 0:
                newAgent = deadAgents.popleft()
                parent = random.choice(list(birthReadyAgents.keys()))
                model_dict[newAgent] = crossover(model_dict[parent], model_dict[newAgent])
                model_dict = simple_mutate(newAgent, model_dict, alpha=0.01)
                model_dict[newAgent].hidden = (torch.zeros(model_dict[newAgent].hidden[0].shape), torch.zeros(model_dict[newAgent].hidden[1].shape))

                agentAges[newAgent] = 0
                agentBirthInterval[newAgent] = 0
                agentBirthInterval[parent] = 0
                eraLogger.births += 1

                try:
                    # Spawn individual in the same place as parent
                    x, y = env.realm.players.entities[parent].pos
                except:
                    # Spawn individual at parent's spawn position
                    x, y = env.realm.players.entities[parent].spawn_pos

                env.realm.players.spawn_individual(x, y, newAgent)

                #parent1_id, parent2_id = random.sample(list(birthReadyAgents.keys()), 2)
            else:
                break

        obs, rewards, dones, infos = env.step(actions)
        step += 1

    #print(f"Era ran for {step - startStep} steps")
    return step, model_dict, eraLogger, longestEra
###


### Training loop.
eraLogger = EraLoggerV2()
#eraLogger.curretEra = 4716 # Training continued
step = 0 # step = 8_900_000 # Training continued from 8.910.117
steps = 10_000_001
longestEra = 100 #longestEra is for recording interesting replays
while step < steps:
    step, model_dict, eraLogger, longestEra = RunEra(
        step, env, model_dict, eraLogger, longestEra)

eraLogger.SaveToFiles(output_dir, EXP_NAME)
pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_final.pickle'),'wb'))