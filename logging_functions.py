import os
import pandas as pd

def calculate_avg_lifetime(env,obs, player_N):
  sum = 0
  for i in range(player_N):
    if i+1 in env.realm.players.entities and i+1 in obs:
      sum += env.realm.players.entities[i+1].__dict__['time_alive'].val
  sum = sum/player_N
  return sum


#EraLogger was changed when trying to fix memory issues. It has been kept because of old files still using it.
class EraLogger:
  def __init__(self):
    self.eraStartStep = 0 #where current era started
    self.exctinct = False
    self.birthTracker = 0
    self.curretEra = 0

  #Given total steps taken by the end of an era and remaining living agents, 
  # stores the length of the era, along with the number of births and living agents.
  def GetCurrentEraData(self, total_steps, living_agents):
    births = self.birthTracker
    eraLength = total_steps - self.eraStartStep

    self.birthTracker = 0
    self.eraStartStep = total_steps
    self.curretEra += 1
    return {
      'births': births,
      'steps': eraLength, 
      'living agents': living_agents,
    }

# def GetAgentXP(env, id):
#   return {
#     'melee': env.realm.players.entities[id].melee_exp.val,
#     'range': env.realm.players.entities[id].range_exp.val,
#     'mage': env.realm.players.entities[id].mage_exp.val,
#     'fishing': env.realm.players.entities[id].fishing_exp.val,
#     'herbalism': env.realm.players.entities[id].herbalism_exp.val,
#     'prospecting': env.realm.players.entities[id].prospecting_exp.val,
#     'carving': env.realm.players.entities[id].carving_exp.val,
#     'alchemy': env.realm.players.entities[id].alchemy_exp.val,
#   }


def GetAgentXP(entity):
  return {
    'melee': entity.melee_exp.val,
    'range': entity.range_exp.val,
    'mage': entity.mage_exp.val,
    'fishing': entity.fishing_exp.val,
    'herbalism': entity.herbalism_exp.val,
    'prospecting': entity.prospecting_exp.val,
    'carving': entity.carving_exp.val,
    'alchemy': entity.alchemy_exp.val,
  }

#def SaveData(outdir, exp_name, unsavedEraData, unsavedAgentEraData, firstSave):
def SaveData(outdir, exp_name, unsavedEraData, firstSave):
  eraFilePath = os.path.join(outdir, exp_name + 'era_data.csv')
  #agentFilePath = os.path.join(outdir, exp_name + 'agent_data.csv')

  if firstSave:
      df = pd.DataFrame(unsavedEraData)
      df.to_csv(eraFilePath, index=False) 
      #df = pd.DataFrame(unsavedAgentEraData)
      #df.to_csv(agentFilePath) 
      firstSave = False
  else:
      df = pd.DataFrame(unsavedEraData)
      df.to_csv(eraFilePath, mode='a', header=False, index=False) 
      #df = pd.DataFrame(unsavedAgentEraData)
      #df.to_csv(agentFilePath, mode='a', header=False) 
  unsavedEraData = []
  #unsavedAgentEraData = []

  #return unsavedEraData, unsavedAgentEraData, firstSave
  return unsavedEraData, firstSave

    # for i in range(player_N):
    #     if i+1 in env.realm.players.entities:
    #         unsavedAgentEraData.append(
    #             GetAgentData(env.realm.players.entities[i+1], i+1)
    #         )


class EraLoggerV2:
  def __init__(self):
    self.firstSave = True
    self.unsavedEraData = []
    self.curretEra = 0
    self.births = 0


  def UpdateEraData(self, eraSteps, living_agents):
    eraData = {
      'era': self.curretEra,
      'births': self.births,
      'steps': eraSteps, 
      'living agents': living_agents,
    }
    self.births = 0
    self.curretEra += 1
    self.unsavedEraData.append(eraData)
  
  def SaveToFiles(self, outdir, exp_name):
    SaveData(outdir, exp_name, self.unsavedEraData, self.firstSave)
    self.firstSave = False
    self.unsavedEraData = []