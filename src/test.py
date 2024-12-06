from hppo import *
# Example usage
observation_dim = 20
discrete_action_dim = 4
continuous_params_dims = [3, 1, 3, 1]  # Continuous parameters for each discrete action

model = HybridActorCritic(observation_dim, discrete_action_dim, continuous_params_dims)
dummy_input = torch.randn(4, observation_dim)  # Batch of 4 observations

discrete_logits, continuous_params, state_value = model(dummy_input)

# Print the outputs
print("Discrete Logits:", discrete_logits.shape)  # Shape: [4, 4]
for i, params in enumerate(continuous_params):
    print(f"Action {i} - Mean: {params['mean'].shape}, Std: {params['std'].shape}")
print("State Value:", state_value.shape)  # Shape: [4, 1]

# Example action sampling
for i in range(4):  # For each sample in the batch
    discrete_action, continuous_action = sample_action(discrete_logits[i], continuous_params)
    print(f"Sampled Discrete Action: {discrete_action}, Continuous Action: {continuous_action}")
