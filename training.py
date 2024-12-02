from hppo import *
import torch
import torch.optim as optim


def train(
        env,
        network,
        optimizer,
        num_episodes=1000,
        gamma=0.99
):
    for episode in range(num_episodes):
        state = env.reset()
        log_probs, values, rewards, dones = [], [], [], []

        done = False
        while not done:
            # Select action
            discrete_action, continuous_action, lob_prob, value = network(state)

            # Perform step in the environment
            action = (discrete_action, continuous_action)
            next_state, reward, done, _ = env.step(action)

            # Store data
            log_probs.append(log_probs)
            values.append(values)
            rewards.append(reward)
            dones.append(done)

            state = next_state

        # Compute loss and update network
        with torch.no_grad:
            _, _, _, next_value = network(torch.FloatTensor(next_state).unsqueeze(0))
        loss = compute_loss(log_probs, values, rewards, dones, next_value, gamma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Loss: {loss.item(): .4f}, Reward: {sum(rewards): .2f}")
