import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#Hidden States
BULL = 'BULL'
BEAR = 'BEAR'
States = [BULL, BEAR]

prior = {BULL:0.5, BEAR:0.5}
likelihood = {BULL:{'UP':0.6, 'DOWN':0.4},
BEAR:{'UP':0.4, 'DOWN':0.6}}

hidden_state = random.choice(States)
print(f"The hidden state of market is {hidden_state}")

def generate_observation(state):
    r = random.random()
    if r < likelihood[state]['UP']:
        return 'UP'
    else:
        return 'DOWN'

def bayesian_update(prior, observation):
    unnormalized_posterior = {}

    for state in States:
        unnormalized_posterior[state] = (
            likelihood[state][observation] * prior[state]
        )

    normalization = sum(unnormalized_posterior.values())

    posterior = {
        state: unnormalized_posterior[state] / normalization
        for state in States
    }

    return posterior
T = 100  # days
beliefs = []
observations = []

current_belief = prior.copy()

for t in range(T):
    obs = generate_observation(hidden_state)
    observations.append(obs)

    current_belief = bayesian_update(current_belief, obs)
    beliefs.append(current_belief[BULL])

plt.figure(figsize=(10, 5))
plt.plot(beliefs, label="P(BULL | data)")
plt.axhline(1.0 if hidden_state == BULL else 0.0,
            color="red", linestyle="--", label="True State")
plt.xlabel("Time")
plt.ylabel("Probability")
plt.title("Bayesian Belief Evolution of Market Regime")
plt.legend()
plt.savefig('test_graph.png')
print("SAVED SUCCESSFULLY")

