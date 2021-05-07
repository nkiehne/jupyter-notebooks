from gym.envs.registration import register
print("YOOO")
register(
    id='LunarLander-hardcore-v2',
    entry_point='gym_lunarlanderhardcore.envs:LunarLanderHardcore',
    #tags={'wrapper_config.TimeLimit.max_episode_steps': 6060},
    #timestep_limit=6060,
    #reward_threshold=1000
)