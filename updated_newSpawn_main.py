import copy
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
EXP_NAME = 'NewSpawn'
startTime = datetime.now()
output_dir = os.path.join(os.path.join('output', startTime.strftime("%m-%d")), EXP_NAME)
os.makedirs(output_dir, exist_ok=True)
###

config, NPCs = set_config()
env = nmmo.Env()  # NOTE, Config changes a ?global nmmo. Previously i env.close() and env = nmmo.Env(), but maybe only resetting the env is fixes memory issues.
player_N = config.PLAYER_N
output_size = 5
output_size_attack = player_N + 1 + NPCs
# Random weights with a FF network
model_dict = {i + 1: PolicyNet(output_size, output_size_attack) for i in
              range(player_N)}  # Dictionary of random models for each agent


### Run one era
def RunEra(startStep, maxStep, env, model_dict, eraLogger):
    step = startStep
    obs = env.reset()

    # Getting spawn positions
    spawn_positions = [(0, 0)]
    for i in range(player_N):
        spawn_positions.append(env.realm.players.entities[i + 1].spawn_pos)

    avail_index = []
    available = False
    life_durations = {i + 1: 0 for i in range(env.config.PLAYER_N)}
    birth_interval = {i + 1: 0 for i in range(env.config.PLAYER_N)}
    mature_age = 50
    interval = 30

    replay_helper = FileReplayHelper()
    env.realm.record_replay(replay_helper)
    replay_helper.reset()

    while step < maxStep:
        if env.num_agents == 0:
            available = True
            if env.num_agents != player_N:
                for i in range(player_N):
                    if i + 1 not in env.realm.players.entities:
                        avail_index.append(i + 1)
            eraLogger.UpdateEraData(step - startStep, len(env.realm.players.entities))
            replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step)), compress=False)
            break
        else:
            available = False

        for i in range(player_N):
            life_durations[i + 1] += 1
            birth_interval[i + 1] += 1


            ##if conditions are right make an offspring
            if len(avail_index)>0 and life_durations[i+1] > mature_age and birth_interval[i+1] > interval and available:
                birth_interval[i + 1] = 0
                new_born = avail_index[0]
                avail_index.pop(0)
                parent = i + 1
                try:
                    # Spawn individual in the same place as parent
                    x, y = env.realm.players.entities[parent].pos
                    # x, y = random.choice(spawn_positions)
                except:
                    # Spawn individual at a random spawn location
                    x, y = random.choice(spawn_positions)
                    spawn_positions[new_born] = (x, y)

                env.realm.players.spawn_individual(x, y, new_born)  ### !CRASHES HERE?

                life_durations[new_born] = 0
                birth_interval[new_born] = 0
                model_dict[new_born] = copy.deepcopy(model_dict[parent])
                model_dict[new_born].hidden = (torch.zeros(model_dict[new_born].hidden[0].shape), torch.zeros(model_dict[new_born].hidden[1].shape))
                simple_mutate(new_born, model_dict, alpha=0.1)
                eraLogger.birthTracker += 1
                break



        if step % 300 == 0:
            print(f'step {step}, era starting step {startStep}')

            # Save era data to file
        if (step + 1) % 10_000 == 0:
            eraLogger.SaveToFiles(output_dir, EXP_NAME)
            replay_helper.save(os.path.join(output_dir, EXP_NAME + str(step)), compress=False)

        actions = {}
        for agent in env.realm.players.entities.keys():
            inp = get_input(env.realm.players.entities[agent], obs[agent]['Tile'], obs[agent]['Entity'],
                            env.realm.players.entities[agent].pos)
            output, style, output_attack = model_dict[agent](inp)
            actions[agent] = {"Move": {"Direction": int(output)},
                              "Attack": {"Style": style, "Target": int(output_attack)}, "Use": {"InventoryItem": 0}}

        obs, rewards, dones, infos = env.step(actions)
        step += 1

    # print(f"Era ran for {step - startStep} steps")
    return step, model_dict, eraLogger


###


### Training loop.
eraLogger = EraLoggerV2()
step = 0
steps = 40_001
while step < steps:
    step, model_dict, eraLogger = RunEra(step, steps, env, model_dict, eraLogger)

pickle.dump(model_dict, open(os.path.join(output_dir, EXP_NAME + '_agents_model_dict_final.pickle'), 'wb'))