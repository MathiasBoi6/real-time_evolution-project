from datetime import datetime
import os
import pickle
import random
from logging_functions import EraLoggerV2
import nmmo
from nmmo.render.replay_helper import FileReplayHelper

from agent_neural_net import PolicyNet, get_input
from config import set_config

import torch

from real_time_evolution import crossover, simple_mutate
torch.set_num_threads(1)


### Save destination.
EXP_NAME = 'DiverseSearchHalfMC'
startTime = datetime.now() 
output_dir = os.path.join(os.path.join('output', startTime.strftime("%m-%d")), EXP_NAME)
os.makedirs(output_dir, exist_ok=True)
###

config, NPCs = set_config()
env = nmmo.Env() #NOTE, Config changes a ?global nmmo. Previously i env.close() and env = nmmo.Env(), but maybe only resetting the env is fixes memory issues.
player_N = config.PLAYER_N
output_size = 5
output_size_attack = player_N+1+NPCs
# Random weights with a FF network
model_dict = {i+1: PolicyNet(output_size, output_size_attack)  for i in range(player_N)} # Dictionary of random models for each agent

### crossover and mutate models
def update_models(model_dict, succededAgents):
    if len(succededAgents) > 1:
        for agent in model_dict.keys():
            if agent not in succededAgents:
                parent1_id, parent2_id = random.sample(succededAgents, 2)
                model_dict[agent] = crossover(model_dict[parent1_id], model_dict[parent2_id])
                model_dict[agent].hidden = (torch.zeros(model_dict[agent].hidden[0].shape), torch.zeros(model_dict[agent].hidden[1].shape))
            model_dict = simple_mutate(agent, model_dict, alpha=0.01)
    else:
        for agent in model_dict.keys():
            if agent not in succededAgents:
                model_dict = simple_mutate(agent, model_dict, alpha=0.01)
    return model_dict

### Run one era
def RunEra(startStep, env, model_dict, agentSuccess, eraLogger):
    step = startStep
    obs = env.reset()
    SUCCEDED = False
    succededAgents = []
    
    replay_helper = FileReplayHelper()
    env.realm.record_replay(replay_helper)
    replay_helper.reset()

    while True:
        if env.num_agents == 0:
            if SUCCEDED:
                model_dict = update_models(model_dict, succededAgents)
            else:
                for agent in model_dict.keys():
                    model_dict = simple_mutate(agent, model_dict, alpha=0.01)
            eraLogger.UpdateEraData(step-startStep, len(env.realm.players.entities))
            break

        if step - startStep >= agentSuccess:
            print("Halfway success")
            model_dict = update_models(model_dict, list(env.realm.players.entities.keys()))
            eraLogger.UpdateEraData(step-startStep, len(env.realm.players.entities))
            replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step) + "_" + str(agentSuccess)), compress=False)
            agentSuccess *= 2
            break

        if step > agentSuccess and not SUCCEDED:
            SUCCEDED = True
            succededAgents = list(env.realm.players.entities.keys())

        if step%300==0:
            print(f'step {step}, maxAge {agentSuccess}, era starting step {startStep}') 

        #Save era data to file
        if (step+1)%10_000 == 0:
            eraLogger.SaveToFiles(output_dir, EXP_NAME)

        actions = {}
        for agent in env.realm.players.entities.keys():
            inp = get_input(env.realm.players.entities[agent], obs[agent]['Tile'], obs[agent]['Entity'], env.realm.players.entities[agent].pos)
            output, style, output_attack = model_dict[agent](inp)
            actions[agent] = {"Move":{"Direction":int(output)} , "Attack":{"Style":style,"Target":int(output_attack)}, "Use":{"InventoryItem":0}}       

        obs, rewards, dones, infos = env.step(actions)
        step += 1

    #print(f"Era ran for {step - startStep} steps")
    return step, model_dict, agentSuccess, eraLogger
###


### Training loop.
eraLogger = EraLoggerV2()
step = 0
steps = 40_001
agentSuccess = 100
while step < steps:
    step, model_dict, agentSuccess, eraLogger = RunEra(
        step, env, model_dict, agentSuccess, eraLogger)

pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_final.pickle'),'wb'))