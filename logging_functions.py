
def calculate_avg_lifetime(env,obs, player_N):
  sum = 0
  for i in range(player_N):
    if i+1 in env.realm.players.entities and i+1 in obs:
      sum += env.realm.players.entities[i+1].__dict__['time_alive'].val
  sum = sum/player_N
  return sum

class EraLogger:
  def __init__(self, startStep):
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
