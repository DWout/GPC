# GPC
Gaussian Process Coach (GPC)

How to set up the environment to test for the Lunar Lander:

1. Install the OpenAI Gym

2. Add the following lines to the __init__ file found in gym/envs/__init__.py:

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

3. Copy and replace the lunar_lander.py file in the gym/envs/box2d folder

4. Run main_example.py


p.s. If you run into any difficulties, try to send me a pm/post issue. Although it does not seem like it, I am still pretty active on git :)
