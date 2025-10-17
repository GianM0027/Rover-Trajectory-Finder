import gc
import os
import json
import torch
import orjson
import numpy as np
from tqdm import tqdm
import gymnasium as gym
from torch import nn, optim

from constants import *
from custom_environment import GridMarsEnv
from experience_manager import ExperienceManager
from impala import ImpalaModel


class Agent:
    def __init__(self,
                 n_environments=None,
                 policy_network=None,
                 dtm_file=None,
                 fov_distance=None,
                 map_size=None,
                 seed=None,
                 max_number_of_steps=None,
                 max_step_height=1,
                 max_drop_height=1):

        self.seed = seed
        self.policy_network = policy_network
        self.max_number_of_steps = max_number_of_steps
        self.max_step_height = max_step_height
        self.max_drop_height = max_drop_height
        self.fov_distance = fov_distance
        self.dtm_file = dtm_file
        self.map_size = map_size
        self.n_environments = n_environments

    def run_simulation(self,
                       max_episodes=None,
                       use_policy_network=False,
                       verbose=False,
                       device="cuda",
                       sample_action=False,
                       render_mode="human",
                       n_actions=8):

        test_environment = GridMarsEnv(dtm=self.dtm_file,
                                       map_size=self.map_size,
                                       fov_distance=self.fov_distance,
                                       rover_max_step=self.max_step_height,
                                       rover_max_drop=self.max_drop_height,
                                       rover_max_number_of_steps=self.max_number_of_steps)
        test_environment.render_mode = render_mode

        if use_policy_network:
            self.policy_network.eval()
            self.policy_network.to(device)

        n_episode = 0
        while True:
            print(f"Episode #{n_episode + 1}")
            observation, _ = test_environment.reset(seed=self.seed)
            terminated = False
            truncated = False

            while not terminated and not truncated:
                if use_policy_network:
                    with torch.no_grad():
                        processed_frame = torch.tensor(observation).float().unsqueeze(0).to(device)
                        action_probs, _ = self.policy_network(processed_frame)

                    if sample_action:
                        action = torch.distributions.Categorical(probs=action_probs).sample().item()
                    else:
                        action = torch.argmax(action_probs).item()
                else:
                    action = np.random.randint(n_actions)

                observation, _, terminated, truncated, _ = test_environment.step(action, verbose=verbose)

            n_episode += 1
            if max_episodes is not None and n_episode >= max_episodes:
                break

    def curriculum_learning_train(self, config, device):
        for step, configuration in config.items():
            print(f"\nTraining {step + 1}/{len(config)} started")
            print(f"The following configuration is being used: {configuration}\n")

            batch_size = self.n_environments * 128
            minibatch_size = batch_size // 8

            training_steps = configuration["training_timesteps"]
            learning_rate = configuration["learning_rate"]
            freeze_cnn = configuration["freeze_cnn"]
            c2 = configuration["c2"]
            weights_to_reload = configuration["weights_to_reload"]

            self.map_size = configuration["map_size"]
            self.fov_distance, self.max_number_of_steps = get_fovDistance_maxSteps(self.map_size)
            self.seed = configuration["training_seed"]

            cnn_weights_path, full_weights_path = get_weights_path(self.map_size, step=step)
            training_info_path, training_losses_path, training_parameters_path = get_training_info_path(self.map_size, step=step)

            self.policy_network = ImpalaModel(input_channels=4)
            optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

            if weights_to_reload is not None:
                self.policy_network(torch.randn(1, 4, self.map_size, self.map_size))

                if "cnn" in weights_to_reload:
                    self.policy_network.load_feature_extractor_weights(weights_to_reload)
                elif "full" in weights_to_reload:
                    self.policy_network.load_full_weights(weights_to_reload)

            if freeze_cnn:
                for name, param in self.policy_network.named_parameters():
                    if name.startswith(('conv_block1', 'conv_block2', 'conv_block3')):
                        param.requires_grad = False

            mars_environments = gym.vector.AsyncVectorEnv([
                lambda: GridMarsEnv(dtm=self.dtm_file,
                                    map_size=self.map_size,
                                    fov_distance=self.fov_distance,
                                    rover_max_step=self.max_step_height,
                                    rover_max_drop=self.max_drop_height,
                                    rover_max_number_of_steps=self.max_number_of_steps)
                for _ in range(self.n_environments)
            ],
                shared_memory=False
            )

            self.train(environments=mars_environments,
                       training_steps=training_steps,
                       batch_size=batch_size,
                       minibatch_size=minibatch_size,
                       epochs=3,
                       cnn_weights_path=cnn_weights_path,
                       full_weights_path=full_weights_path,
                       training_info_path=training_info_path,
                       training_losses_path=training_losses_path,
                       training_parameters_path=training_parameters_path,
                       device=device,
                       learning_rate=learning_rate,
                       save_interval=100000,
                       c2=c2,
                       optimizer=optimizer)

            # free GPU memory here to avoid pytorch errors
            del self.policy_network
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
            mars_environments.close()

    def train(self,
              environments,
              training_steps=1000000,
              batch_size=512,
              minibatch_size=16,
              epochs=1,
              device='cuda',
              clip_ratio=0.2,
              c1=0.5,
              c2=0.01,
              learning_rate=1e-5,
              cnn_weights_path=None,
              full_weights_path=None,
              training_info_path=None,
              training_losses_path=None,
              training_parameters_path=None,
              save_interval=10000,
              optimizer=None):

        mse_loss = nn.MSELoss()
        if optimizer is None:
            optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.policy_network.train()
        self.policy_network.to(device)

        experience_manager = ExperienceManager(batch_size=batch_size,
                                               minibatch_size=minibatch_size,
                                               n_envs=self.n_environments)
        all_episodes_info = []
        ppo_losses = []

        if training_parameters_path:
            parent_dir = os.path.dirname(training_parameters_path)
            os.makedirs(parent_dir, exist_ok=True)

            training_parameters = {
                "map_size": self.map_size,
                "fov_distance": self.fov_distance,
                "max_number_of_steps": self.max_number_of_steps,
                "max_step_height": self.max_step_height,
                "max_drop_height": self.max_drop_height,
                "map_name": self.dtm_file.img_path,

                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "minibatch_size": minibatch_size,
                "epochs": epochs,
                "clip_ratio": clip_ratio,
                "c1": c1,
                "c2": c2,
            }

            print(f"Saving training parameters to {training_parameters_path}")
            with open(training_parameters_path, 'w') as f:
                json.dump(training_parameters, f, indent=4)

        info_to_save = {
            env: self._init_info_to_save() for env in range(self.n_environments)
        }

        # reset environment and save initial position of agent during first episode
        observations, info = environments.reset(seed=self.seed)
        for env in range(self.n_environments):
            info_to_save[env]["agent_positions"].append(info["agent_relative_position"][env].tolist())

        current_step = 0
        with tqdm(total=training_steps) as pbar:
            while current_step < training_steps:
                current_step += self.n_environments
                pbar.update(self.n_environments)

                with torch.no_grad():
                    obs_tensor = torch.tensor(observations).to(device)
                    action_probs, values = self.policy_network(obs_tensor)

                dist = torch.distributions.Categorical(probs=action_probs)
                actions = dist.sample()
                prob_of_taken_action = action_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                observations, rewards, terminated, truncated, info = environments.step(actions.cpu().numpy())

                experience_manager.appendTrajectory(
                    states=observations,
                    actions=actions.cpu().numpy(),
                    action_probs=prob_of_taken_action.detach().cpu().numpy(),
                    rewards=rewards,
                    values=values.squeeze(-1).cpu().numpy(),
                    terminated=terminated,
                    truncated=truncated
                )

                for env in range(self.n_environments):
                    info_to_save[env]["episode_length"] += 1
                    info_to_save[env]["agent_positions"].append(info["agent_relative_position"][env].tolist())
                    info_to_save[env]["rewards"].append(rewards[env].item())

                    if terminated[env] or truncated[env]:
                        info_to_save[env]["terminated"] = bool(terminated[env])
                        info_to_save[env]["truncated"] = bool(truncated[env])
                        info_to_save[env]["target_position"] = info["target_position"][env].tolist()
                        info_to_save[env]["local_map"] = info["local_map"][env].tolist()

                        all_episodes_info.append(info_to_save[env])
                        info_to_save[env] = self._init_info_to_save()

                if experience_manager.is_full():
                    with torch.no_grad():
                        obs_tensor = torch.tensor(observations).to(device)
                        _, next_values = self.policy_network(obs_tensor)

                    dataloader = experience_manager.get_batches(next_values=next_values.cpu().numpy(), device=device)
                    update_losses = []
                    for _ in range(epochs):
                        for states, actions, old_action_probs, advantages, returns in dataloader:
                            current_action_dist, values = self.policy_network(states)
                            current_action_probs = current_action_dist.gather(
                                dim=1,
                                index=actions.unsqueeze(-1).to(torch.int64)
                            ).squeeze(-1)

                            prob_ratio = current_action_probs / (old_action_probs + 1e-10)
                            clipped_ratio = torch.clamp(prob_ratio, 1 - clip_ratio, 1 + clip_ratio)

                            actor_loss = -torch.min(prob_ratio * advantages, clipped_ratio * advantages).mean()
                            critic_loss = mse_loss(values.squeeze(-1), returns)
                            entropy = -(current_action_dist * torch.log(current_action_dist + 1e-10)).mean()

                            total_loss = actor_loss + (c1 * critic_loss) - (c2 * entropy)
                            update_losses.append(total_loss.item())

                            optimizer.zero_grad()
                            total_loss.backward()
                            # torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=0.5)
                            optimizer.step()

                    experience_manager.clear()
                    ppo_losses.append(sum(update_losses) / len(update_losses))

                if current_step % save_interval < self.n_environments or current_step >= training_steps:
                    self._save_training_info(training_losses_path, ppo_losses, training_info_path, all_episodes_info,
                                             cnn_weights_path, full_weights_path)

    @classmethod
    def _init_info_to_save(cls):
        return {
            "agent_positions": [],
            "rewards": [],
            "terminated": False,
            "truncated": False,
            "episode_length": 0,
            "target_position": None,
            "local_map": None
        }

    def _save_training_info(
            self,
            training_losses_path,
            ppo_losses,
            training_info_path,
            all_episodes_info,
            cnn_weights_path,
            full_weights_path
    ):

        def save_json(data, path):
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(path, "wb") as f:
                f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

        # Save PPO training losses
        if training_losses_path:
            save_json(ppo_losses, training_losses_path)

        # Save episode-level training info
        if training_info_path:
            save_json(all_episodes_info, training_info_path)

        # Save network weights
        if cnn_weights_path and full_weights_path:
            os.makedirs(os.path.dirname(cnn_weights_path), exist_ok=True)
            os.makedirs(os.path.dirname(full_weights_path), exist_ok=True)
            self.policy_network.save_weights(cnn_weights_path, full_weights_path)
