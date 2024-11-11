import nmmo


def set_config():
    config = nmmo.config.Default

    # Define the amount of resources on the map
    nmmo.config.Default.MAP_CENTER=32
    nmmo.config.Default.PROGRESSION_SPAWN_CLUSTERS=4
    nmmo.config.Default.PROGRESSION_SPAWN_UNIFORMS=8

    # Define the basic things
    nmmo.config.Default.TERRAIN_WATER = 0.5
    nmmo.config.Default.TERRAIN_DISABLE_STONE = True
    nmmo.config.Default.TERRAIN_GRASS = 0.6
    nmmo.config.Default.TERRAIN_FOILAGE = 0.5
    nmmo.config.Default.TERRAIN_FLIP_SEED = True
    nmmo.config.Default.HORIZON = 2**15-1

    # Remove the death fog
    nmmo.config.Default.PLAYER_DEATH_FOG_FINAL_SIZE = 0
    nmmo.config.Default.PLAYER_DEATH_FOG_SPEED = 0


    ##Disable system modes
    nmmo.config.Default.COMBAT_SYSTEM_ENABLED = True
    nmmo.config.Default.EXCHANGE_SYSTEM_ENABLED = False
    nmmo.config.Default.COMMUNICATION_SYSTEM_ENABLED = False
    nmmo.config.Default.PROFESSION_SYSTEM_ENABLED = True
    nmmo.config.Default.PROGRESSION_SYSTEM_ENABLED = True
    nmmo.config.Default.EQUIPMENT_SYSTEM_ENABLED = True
    nmmo.config.Default.NPC_SYSTEM_ENABLED = True
    nmmo.config.Default.COMBAT_SPAWN_IMMUNITY = 0
    nmmo.config.Default.COMBAT_MELEE_REACH = 3
    nmmo.config.Default.COMBAT_RANGE_REACH = 3
    nmmo.config.Default.COMBAT_MAGE_REACH = 3

    ##Population Size
    nmmo.config.Default.PLAYER_N = 64
    nmmo.config.Default.NPC_N = 0
    NPCs = nmmo.config.Default.NPC_N

    ##Player Input
    nmmo.config.Default.PLAYER_N_OBS = 25
    nmmo.config.Default.PLAYER_VISION_RADIUS = 7

    return config, NPCs