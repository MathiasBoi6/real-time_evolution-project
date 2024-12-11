# Create 1 random CNN and share it between all agents. With the same CNN, the dense layers below are more coherent when swapped between agents.


from collections import deque
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

from real_time_evolution import crossover
torch.set_num_threads(1)


### Save destination.
EXP_NAME = 'RandomVision'
startTime = datetime.now() 
output_dir = os.path.join(os.path.join('output', startTime.strftime("%m-%d")), EXP_NAME)
os.makedirs(output_dir, exist_ok=True)
MATURE_AGE = 50
INTERVAL = 30
###

config, NPCs = set_config()

nmmo.config.Default.RESOURCE_FOILAGE_RESPAWN /= 4

env = nmmo.Env() #NOTE, Config changes a ?global nmmo. Previously i env.close() and env = nmmo.Env(), but maybe only resetting the env is fixes memory issues.
player_N = config.PLAYER_N
output_size = 5
output_size_attack = player_N+1+NPCs
# Random weights with a FF network

def getModels():
    with torch.no_grad():
        baseModel = PolicyNet(output_size, output_size_attack)
        model_dict = {i+1: PolicyNet(output_size, output_size_attack)  for i in range(player_N)} # Dictionary of random models for each agent
        for model in model_dict.keys():
            model_dict[model].tile_conv = baseModel.tile_conv
    return model_dict
model_dict = getModels()
# Continue training from a previous model
#model_dict = pickle.load(open(EXP_NAME+'_agents_model_dict.pickle','rb'))

#does not mutate convolutional filter
def biased_mutate(player_num, model_dict, alpha=0.01):
  for name, param in model_dict[player_num].named_parameters():
    if "tile_conv" not in name:
        with torch.no_grad():
            param.add_(torch.randn(param.size()) * alpha)
  return model_dict


### Run one era
def RunEra(startStep, env, model_dict, eraLogger, longestEra):
    step = startStep
    obs = env.reset()

    replay_helper = FileReplayHelper()
    env.realm.record_replay(replay_helper)
    replay_helper.reset()

    while True:
        if env.num_agents <= int(64 * 0.1):
            print("Reset")
            for agent in model_dict.keys():
                if agent not in list(env.realm.players.entities.keys()): 
                    parent1_id = random.sample(list(env.realm.players.entities.keys()), 1)[0]
                    model_dict[agent] = crossover(model_dict[parent1_id], model_dict[agent])
                    model_dict = biased_mutate(agent, model_dict, alpha=0.01)
                model_dict[agent].hidden = (torch.zeros(model_dict[agent].hidden[0].shape), torch.zeros(model_dict[agent].hidden[1].shape))
            eraLogger.UpdateEraData(step-startStep, len(env.realm.players.entities))

            if (step - startStep) > longestEra:
                replay_helper.save(os.path.join(output_dir, EXP_NAME + str(int(longestEra))), compress=False)
                longestEra *= 1.2
                longestEra = int(longestEra)
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

        obs, rewards, dones, infos = env.step(actions)
        step += 1

    #print(f"Era ran for {step - startStep} steps")
    return step, model_dict,  eraLogger, longestEra
###


### Training loop.
eraLogger = EraLoggerV2()
step = 0 
steps = 10_000_001
longestEra = 70
while step < steps:
    step, model_dict, eraLogger, longestEra = RunEra(
        step, env, model_dict, eraLogger, longestEra)

eraLogger.SaveToFiles(output_dir, EXP_NAME)
pickle.dump(model_dict,open(os.path.join(output_dir, EXP_NAME+'_agents_model_dict_final.pickle'),'wb'))