import os
import glob
import torch
import numpy as np
from omegaconf import OmegaConf
import argparse
import pygame
import sys

# Initialize Pygame font module
pygame.font.init()

# Add a try-except block for the import to provide a helpful message
try:
    import imageio
except ImportError:
    print("Module 'imageio' not found. Please install it with 'pip install imageio imageio-ffmpeg' to generate videos.")
    imageio = None

from Game import SnakeEnv
from brain import actor

# --- Visualization constants and helpers ---
# Colors for visualization
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
BLUE = (0, 0, 255)

# --- Font handling ---
# Define a dummy font class for fallback in case of errors
class DummyFont:
    def render(self, *args, **kwargs):
        return pygame.Surface((0,0))

# Initialize font variables. They will be loaded inside generate_video.
TITLE_FONT, ACTION_FONT_BOLD, ACTION_FONT = None, None, None


def draw_grid_visualization(surface, state, M, direction):
    """Draws the agent's grid-based POV, rotated to the snake's perspective."""
    pov_size = surface.get_width()
    cell_size = pov_size / M
    grid = state.reshape((M, M))

    # Rotate grid so 'up' on the POV display is always the snake's forward direction
    if direction == 'DOWN':
        grid = np.rot90(grid, k=2)  # 180 deg
    elif direction == 'RIGHT':
        grid = np.rot90(grid, k=1)  # 90 deg CCW
    elif direction == 'LEFT':
        grid = np.rot90(grid, k=-1) # 90 deg CW
    # 'UP' requires no rotation

    for r in range(M):
        for c in range(M):
            val = grid[r, c]
            color = GRAY
            if val == 1:   # Food
                color = GREEN
            elif val == -1: # Wall/Body
                color = RED
            
            pygame.draw.rect(surface, color, (c * cell_size, r * cell_size, cell_size, cell_size))
            pygame.draw.rect(surface, WHITE, (c * cell_size, r * cell_size, cell_size, cell_size), 1)

    # Draw agent in the center, always pointing up
    center_r, center_c = M // 2, M // 2
    agent_rect = pygame.Rect(center_c * cell_size, center_r * cell_size, cell_size, cell_size)
    pygame.draw.rect(surface, BLUE, agent_rect)

def draw_raycast_visualization(surface, state, K):
    """Draws the agent's raycast-based POV."""
    pov_size = surface.get_width()
    s_rays, r_rays, l_rays = state[0:K], state[K:2*K], state[2*K:3*K]
    agent_pos = (pov_size / 2, pov_size / 2)
    ray_len_per_step = (pov_size * 0.45) / K
    colors = {0.0: GRAY, 1.0: GREEN, -1.0: RED}

    # Straight rays (up), Right turn rays (right), Left turn rays (left)
    ray_directions = [(s_rays, (0, -1)), (r_rays, (1, 0)), (l_rays, (-1, 0))]

    for rays, (dx, dy) in ray_directions:
        for i, val in enumerate(rays):
            start_pos = (agent_pos[0] + dx * i * ray_len_per_step, agent_pos[1] + dy * i * ray_len_per_step)
            end_pos = (agent_pos[0] + dx * (i + 1) * ray_len_per_step, agent_pos[1] + dy * (i + 1) * ray_len_per_step)
            pygame.draw.line(surface, colors.get(val, BLACK), start_pos, end_pos, 5)

    # Draw agent body and direction arrow
    pygame.draw.circle(surface, BLUE, (int(agent_pos[0]), int(agent_pos[1])), 10)
    pygame.draw.line(surface, BLUE, agent_pos, (agent_pos[0], agent_pos[1] - 15), 3)

def draw_network_visualization(surface, action_probs, chosen_action):
    """Draws a visualization of the network's output probabilities."""
    surface.fill(BLACK)
    width, height = surface.get_width(), surface.get_height()

    # --- Title ---
    title_surface = TITLE_FONT.render("Actor Output", True, WHITE)
    title_rect = title_surface.get_rect(center=(width / 2, 20))
    surface.blit(title_surface, title_rect)

    # --- Setup ---
    probs = action_probs.squeeze().tolist()
    action_labels = ["Straight", "Right", "Left"]
    bar_color = GRAY
    highlight_color = GREEN

    # --- Drawing Layout ---
    num_actions = len(probs)
    content_area_y_start = 45
    content_area_height = height - content_area_y_start - 15
    
    bar_spacing = content_area_height / num_actions
    bar_height = bar_spacing * 0.6  # 60% of the available vertical space for the bar itself
    
    left_margin = 10
    label_width = 80
    max_bar_width = width - label_width - left_margin - 50 # Space for text

    for i, prob in enumerate(probs):
        is_chosen = (i == chosen_action)
        
        color = highlight_color if is_chosen else bar_color
        font = ACTION_FONT_BOLD if is_chosen else ACTION_FONT

        # Y position for the top of the bar
        y_pos = content_area_y_start + (i * bar_spacing) + (bar_spacing - bar_height) / 2

        # Draw action label
        label_surface = font.render(action_labels[i], True, color)
        surface.blit(label_surface, (left_margin, y_pos + (bar_height - label_surface.get_height()) / 2))

        # Draw probability bar background (for max-length reference)
        bg_bar_rect = pygame.Rect(left_margin + label_width, y_pos, max_bar_width, bar_height)
        pygame.draw.rect(surface, (30, 30, 30), bg_bar_rect)

        # Draw probability bar
        bar_width = prob * max_bar_width
        bar_rect = pygame.Rect(left_margin + label_width, y_pos, bar_width, bar_height)
        pygame.draw.rect(surface, color, bar_rect)

        # Draw probability text
        prob_text = f"{prob:.1%}"
        prob_surface = font.render(prob_text, True, WHITE)
        surface.blit(prob_surface, (left_margin + label_width + 5, y_pos + (bar_height - prob_surface.get_height()) / 2))

def get_model_input_size(config):
    """Helper function to determine the model's input size from the config."""
    state_type = config.run_parameters.state_type
    state_size = config.run_parameters.state_size
    if state_type == "raycast":
        return state_size * 3
    elif state_type == "grid":
        return state_size ** 2
    else: # vector
        return 12

def generate_video(config_path, actor_path, output_path, seed):
    """
    Generates a video of a Snake game episode using a trained agent.

    Args:
        config_path (str): Path to the configuration YAML file.
        actor_path (str): Path to the saved actor model (.pth file).
        output_path (str): Path to save the generated video (.mp4).
        seed (int): The seed for the environment.
    """
    if imageio is None:
        print("Cannot generate video, 'imageio' is not installed.")
        return

    config = OmegaConf.load(config_path)
    
    # --- Initialize Environment ---
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

    env = SnakeEnv(
        render_mode='rgb_array',
        state_type=config.run_parameters.state_type,
        M=config.run_parameters.state_size if config.run_parameters.state_type == 'grid' else 5,
        K=config.run_parameters.state_size if config.run_parameters.state_type == 'raycast' else 3,
        **env_params
    )

    # --- Initialize/Re-initialize fonts ---
    # This needs to be done for each video because env.close() calls pygame.quit(),
    # which invalidates any previously created font objects.
    global TITLE_FONT, ACTION_FONT_BOLD, ACTION_FONT
    try:
        TITLE_FONT = pygame.font.Font(None, 30)
        ACTION_FONT_BOLD = pygame.font.Font(None, 28)
        ACTION_FONT = pygame.font.Font(None, 24)
    except Exception as e:
        # This can happen if the pygame instance from the previous video was not quit properly
        # or on the first run if there's a system issue.
        print(f"Warning: Could not initialize fonts, falling back to dummy fonts. Error: {e}")
        TITLE_FONT = ACTION_FONT_BOLD = ACTION_FONT = DummyFont()

    # --- Initialize Agent ---
    input_size = get_model_input_size(config)
    agent = actor(
        input_size=input_size, 
        output_size=3, 
        hidden_size=config.run_parameters.hidden_size, 
        seed=config.seed
    )
    agent.load_state_dict(torch.load(actor_path))
    agent.eval() # Set agent to evaluation mode

    # --- Setup for combined video frame with POV ---
    pov_size = env.h  # Make POV display square with height of game screen
    pov_surface = pygame.Surface((pov_size, pov_size))
    network_surface = pygame.Surface((pov_size, pov_size))

    # --- Run Episode and Record Frames ---
    frames = []
    state, _ = env.reset(seed=seed)
    done = False
    
    while not done:
        # Render game screen
        game_frame = env.render()
        
        # Render agent's POV on a separate surface
        pov_surface.fill(BLACK)
        state_type = config.run_parameters.state_type
        if state_type == "grid":
            draw_grid_visualization(
                pov_surface, state, config.run_parameters.state_size, env.direction
            )
        elif state_type == "raycast":
            draw_raycast_visualization(
                pov_surface, state, config.run_parameters.state_size
            )
        
        # Convert POV surface to numpy array
        pov_frame = pygame.surfarray.array3d(pov_surface)
        pov_frame = np.transpose(pov_frame, (1, 0, 2))

        # Agent determines next action based on the current state
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_probs = agent(state_tensor)
        action = torch.argmax(action_probs, dim=1).item()
        
        # Render network visualization
        draw_network_visualization(network_surface, action_probs, action)
        network_frame = pygame.surfarray.array3d(network_surface)
        network_frame = np.transpose(network_frame, (1, 0, 2))

        # Combine game, POV, and network frames side-by-side and append
        combined_frame = np.hstack((game_frame, pov_frame, network_frame))

        # Pad frame to be divisible by 16 for video codecs to prevent resizing warnings.
        # Most codecs work with blocks of 16x16 pixels (macroblocks).
        h, w, _ = combined_frame.shape
        macro_block_size = 16
        pad_h = (macro_block_size - h % macro_block_size) % macro_block_size
        pad_w = (macro_block_size - w % macro_block_size) % macro_block_size

        padded_frame = np.pad(
            combined_frame,
            ((0, pad_h), (0, pad_w), (0, 0)),
            mode='constant',
            constant_values=0  # Pad with black
        )
        frames.append(padded_frame)

        # Take step in environment to get the next state
        state, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()

    # --- Save Video ---
    imageio.mimsave(output_path, frames, fps=15)
    print(f"  Successfully saved video to {output_path}")


def generate_videos_for_folder(folder_path):
    """
    Generates videos for all actor models found in a specific folder.
    It uses the config.yaml within that folder.
    """
    config_path = os.path.join(folder_path, 'config.yaml')
    if not os.path.exists(config_path):
        print(f"Warning: No 'config.yaml' found in {folder_path}. Skipping.")
        return

    config = OmegaConf.load(config_path)
    actor_paths = glob.glob(os.path.join(folder_path, '*.pth'))
    
    if not actor_paths:
        print(f"No actor models (.pth files) found in {folder_path}.")
        return

    print(f"Found {len(actor_paths)} actor(s) in {folder_path}.")
    for actor_path in actor_paths:
        print(f"Processing actor: {os.path.basename(actor_path)}")
        for seed in config.save_parameters.video_seeds:
            output_path = actor_path.replace('.pth', f'_seed{seed}.mp4')
            generate_video(config_path, actor_path, output_path, seed)


def generate_missing_videos(base_folder="raw_models"):
    """
    Traverses all subfolders of a base directory, finds actor models,
    and generates videos if they don't already exist.
    """
    for root, _, files in os.walk(base_folder):
        actor_paths = [os.path.join(root, f) for f in files if f.endswith('.pth')]
        config_path = os.path.join(root, 'config.yaml')

        if not actor_paths or not os.path.exists(config_path):
            continue

        print(f"\nChecking folder: {root}")
        config = OmegaConf.load(config_path)
        
        for actor_path in actor_paths:
            for seed in config.save_parameters.video_seeds:
                video_path = actor_path.replace('.pth', f'_seed{seed}.mp4')
                if not os.path.exists(video_path):
                    print(f"Missing video for {os.path.basename(actor_path)} with seed {seed}. Generating...")
                    try:
                        generate_video(config_path, actor_path, video_path, seed)
                    except Exception as e:
                        print(f"  Failed to generate video {os.path.basename(video_path)}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate videos for trained Snake agents.")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='missing', 
        choices=['missing', 'folder'],
        help="Operation mode: 'missing' to find and generate all missing videos (default), 'folder' to generate for a specific folder."
    )
    parser.add_argument(
        '--folder', 
        type=str, 
        default=None,
        help="Path to the specific run folder (required for 'folder' mode)."
    )
    args = parser.parse_args()

    if imageio is None:
        sys.exit(1) # Exit if imageio is not available

    if args.mode == 'missing':
        print("--- Searching for and generating all missing videos ---")
        generate_missing_videos()
        print("\n--- Finished ---")
    elif args.mode == 'folder':
        if args.folder is None:
            print("Error: --folder argument is required for 'folder' mode.")
            sys.exit(1)
        if not os.path.isdir(args.folder):
            print(f"Error: Folder not found at '{args.folder}'")
            sys.exit(1)
            
        print(f"--- Generating videos for folder: {args.folder} ---")
        generate_videos_for_folder(args.folder)
        print("\n--- Finished ---")
