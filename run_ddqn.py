from lib.agent import DDQNAgent

from lib.plot_q_value import plot_state_q

SAVE_NAME = "DDQN"

agent = DDQNAgent(render_mode=None)
agent.lr = 0.005
agent.build()

agent.train(1000)

agent.save(SAVE_NAME)
agent.log(f"{SAVE_NAME}-LOG")

plot_state_q(agent.Q, agent.discrete_action_space_actual, episode=1000)

agent.plot("Performance")
