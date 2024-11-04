
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

  def SaveCurrentEraData(self, total_steps, living_agents):
    births = self.birthTracker
    eraLength = total_steps - self.eraStartStep

    self.birthTracker = 0
    self.eraStartStep = total_steps
    return {
      'births': births,
      'steps': eraLength, 
      'living agents': living_agents,
    }
  
