import os
import sys

# Get the directory of the current script (train.py)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory of the script_dir, which is 'd:\python\RL_games'
project_root = os.path.dirname(script_dir)
# Insert the project root to the beginning of sys.path to make 'Snake' discoverable as a package.
sys.path.insert(0, project_root)

from omegaconf import OmegaConf
from Snake.Game import SnakeEnv
import Snake.brain as brain
import tqdm, time, glob
import torch
from torch.distributions import Categorical
# Add a try-except block for the import to provide a helpful message
try:
    from Snake.generate_videos import generate_video
except ImportError:
    print("Warning: 'generate_videos.py' not found. End-of-training video generation will be skipped.")
    generate_video = None

# Add a try-except block for matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: 'matplotlib' not found. Performance plot will not be generated.")
    plt = None

config = OmegaConf.load("Snake/config.yaml")

def run_episode(env, agent, device, show_progress=True):
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    rewards = []
    log_probs = []
    states = []
    action_probabilities = []
    with tqdm.tqdm(desc=f"Episode progress", leave=False, unit=" step",
                   disable=not show_progress) as pbar:
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Add batch dimension
            action_probs = agent(state_tensor)
            m = Categorical(action_probs)
            action = m.sample()
            log_prob = m.log_prob(action)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            total_reward += reward
            rewards.append(reward)
            log_probs.append(log_prob)
            action_probabilities.append(action_probs)
            states.append(state)
            state = next_state
            done = terminated or truncated

            pbar.update(1)
            pbar.set_postfix(curr_rew=total_reward)

    # Get final snake length at the end of the episode
    snake_length = len(env.snake)
    return total_reward, rewards, log_probs, states, action_probabilities, snake_length

def compute_returns(rewards, device):

    # Compute discounted rewards
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + config.run_parameters.gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32).to(device)

    # Normalize rewards
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    return discounted_rewards

def calculate_policy_loss(log_probs, discounted_rewards, action_probabilities, entropy_coeff):
    """
    Calculates the combined loss for a single episode.
    This includes the policy gradient loss and an entropy bonus to encourage exploration.
    """
    policy_loss_terms = []
    entropy_terms = []
    for log_prob, R, action_prob in zip(log_probs, discounted_rewards, action_probabilities):
        policy_loss_terms.append(-log_prob * R)
        # Entropy of the policy distribution is a measure of its randomness.
        # H(pi) = - sum(pi(a|s) * log(pi(a|s)))
        # We want to maximize entropy, which is equivalent to minimizing -H(pi).
        entropy_terms.append(Categorical(probs=action_prob).entropy())

    policy_loss = torch.cat(policy_loss_terms).sum()
    entropy_bonus = torch.cat(entropy_terms).sum()
    return policy_loss - entropy_coeff * entropy_bonus

def plot_performance(episode_rewards, save_path):
    """Saves a plot of episode rewards over time."""
    if plt is None:
        print("Cannot plot performance, 'matplotlib' is not installed.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(episode_rewards)
    plt.title('Agent Performance: Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Performance plot saved to '{save_path}'")

def plot_snake_length(episode_lengths, save_path):
    """Saves a plot of snake length over time."""
    if plt is None:
        print("Cannot plot snake length, 'matplotlib' is not installed.")
        return
    plt.figure(figsize=(12, 6))
    plt.plot(episode_lengths)
    plt.title('Agent Performance: Final Snake Length per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Final Snake Length')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"Snake length plot saved to '{save_path}'")

def train():
    print('training version:', config.project.version)
    # Train the agent using REINFORCE algorithm
    if config.run_parameters.run_mode != 'training':
        print(f"Run mode is '{config.run_parameters.run_mode}', skipping training.")
        return

    # Create save directory
    save_dir = os.path.join("raw_models", config.project.version, config.save_parameters.run_name)
    os.makedirs(save_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(save_dir, 'config.yaml'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(config.seed)
    start_time = time.time()
    episode_rewards = []
    episode_snake_lengths = []

    # Prepare environment parameters
    env_config = config.env_parameters
    env_params = {
        "reward_food": env_config.reward_values.food,
        "reward_death": env_config.reward_values.death,
        "reward_step": 0,
        "max_steps_multiplier": env_config.max_steps_multiplier
    }
    if env_config.reward_type == 'dense':
        env_params["reward_step"] = env_config.reward_values.step
    env_params["grid_w"] = env_config.get("w", 32)
    env_params["grid_h"] = env_config.get("h", 24)
    # initialize environment
    match config.run_parameters.state_type:
        case "raycast":
            env = SnakeEnv(state_type='raycast', K=config.run_parameters.state_size, **env_params)
            # 3 one-hot encoded classes for each of the K*3 rays
            input_size=config.run_parameters.state_size*3*3
        case "grid":
            env = SnakeEnv(state_type='grid', M=config.run_parameters.state_size, **env_params)
            # 3 one-hot encoded classes for each cell in the M*M grid
            input_size=config.run_parameters.state_size**2*3
        case _:
            env = SnakeEnv(state_type='vector', **env_params)
            input_size=12

    match config.network_parameters.network_type:
        case "mlp":
            agent = brain.actor_mlp(input_size=input_size, output_size=3, hidden_size=config.run_parameters.hidden_size, seed=config.seed)
        case "cnn":
            # Determine the original 2D shape of the state for the CNN
            if config.run_parameters.state_type == 'grid':
                # Shape is (Channels, Height, Width) -> (3, M, M)
                state_size = config.run_parameters.state_size
                state_shape = (3, state_size, state_size)
            elif config.run_parameters.state_type == 'raycast':
                # Shape is (Channels, Height, Width) -> (3 channels, 3 directions, K steps)
                state_size = config.run_parameters.state_size
                state_shape = (3, 3, state_size)
            else:
                raise ValueError(f"CNN network type is not supported for state_type '{config.run_parameters.state_type}'")

            agent = brain.actor_cnn(input_size=input_size, output_size=3, mlp_head_size=config.network_parameters.mlp_head_size,
                                    kernel_sizes=config.network_parameters.kernel_sizes, cnn_filters=config.network_parameters.cnn_filters,
                                    seed=config.seed, state_shape=state_shape)
        case _:
            raise ValueError(f"Unsupported network type: {config.network_parameters.network_type}")
    agent.to(device)
    
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.run_parameters.learning_rate)

    # --- Batching setup ---
    batch_size = config.run_parameters.batch_size
    accumulated_loss = 0.0
    batch_episode_count = 0
    optimizer.zero_grad()

    pbar = tqdm.tqdm(range(config.run_parameters.n_episodes), desc="Training Snake", position=0)

    for episode in pbar:
        total_reward, rewards, log_probs, _, action_probs_list, snake_length = run_episode(
            env, agent, device, show_progress=config.run_parameters.show_episode_progress
        )
        episode_rewards.append(total_reward)
        episode_snake_lengths.append(snake_length)

        discounted_rewards = compute_returns(rewards, device)

        # Calculate loss for this episode and accumulate
        episode_loss = calculate_policy_loss(
            log_probs,
            discounted_rewards,
            action_probs_list,
            config.run_parameters.entropy_coeff)
        accumulated_loss += episode_loss
        batch_episode_count += 1

        # If batch is complete, update the policy
        if batch_episode_count == batch_size:
            # Average the loss over the batch
            avg_loss = accumulated_loss / batch_episode_count
            
            # Backpropagate and update weights
            avg_loss.backward()
            optimizer .step()
            
            # Reset for the next batch
            optimizer.zero_grad()
            accumulated_loss = 0.0
            batch_episode_count = 0

        # Periodic model saving
        if (episode + 1) % config.save_parameters.save_actor_frequency == 0:
            save_path = os.path.join(save_dir, f"{config.save_parameters.run_name}_ep{episode + 1}.pth")
            torch.save(agent.state_dict(), save_path)

        pbar.set_postfix({
            "reward": f"{total_reward:.2f}",
            "length": snake_length,
        })

    # After the loop, update with any remaining episodes in the last batch
    if batch_episode_count > 0:
        print(f"\nPerforming final update on remaining {batch_episode_count} episodes.")
        avg_loss = accumulated_loss / batch_episode_count
        avg_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save final model
    final_save_path = os.path.join(save_dir, f"{config.save_parameters.run_name}_final.pth")
    torch.save(agent.state_dict(), final_save_path)
    print(f"\nTraining finished. Final model saved to '{final_save_path}'")

    # Plot performance graph
    if plt:
        plot_path = os.path.join(save_dir, f"{config.save_parameters.run_name}_performance.png")
        plot_performance(episode_rewards, plot_path)

        length_plot_path = os.path.join(save_dir, f"{config.save_parameters.run_name}_snake_length.png")
        plot_snake_length(episode_snake_lengths, length_plot_path)

    # Generate end-of-training videos for all saved models
    global generate_video
    if generate_video:
        print("\nGenerating end-of-training videos for all saved models...")
        config_path = os.path.join(save_dir, 'config.yaml')

        # Find all saved actor models
        actor_paths = glob.glob(os.path.join(save_dir, '*.pth'))
        if not actor_paths:
            print("No actor models found to generate videos for.")

        for actor_path in actor_paths:
            print(f"\nProcessing actor: {os.path.basename(actor_path)}")
            for seed in config.save_parameters.video_seeds:
                video_path = actor_path.replace('.pth', f'_seed{seed}.mp4')
                print(f"  Generating video for seed {seed}...")
                try:
                    generate_video(config_path=config_path, actor_path=actor_path, output_path=video_path, seed=seed)
                except Exception as e:
                    if "imageio" in str(e):
                        print("  Skipping video generation: 'imageio' or 'imageio-ffmpeg' not installed. Please run 'pip install imageio imageio-ffmpeg'")
                        generate_video = None # Prevent further attempts
                        break # Stop trying for this actor
                    print(f"  Could not generate video for seed {seed}: {e}")
            if generate_video is None:
                break # Stop trying to generate for other actors
        print("\nVideo generation complete.")

if __name__ == "__main__":
    train()