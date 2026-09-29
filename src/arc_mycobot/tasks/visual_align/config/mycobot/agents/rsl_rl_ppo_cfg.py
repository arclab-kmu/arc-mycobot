"""PPO configuration for the marked-cube visual alignment baseline."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class MyCobotVisualAlignPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 300
    save_interval = 25
    experiment_name = "visual_align_mycobot_280_jn_moving_box"
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[128, 128],
        critic_hidden_dims=[128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class MyCobotYoloHandPPORunnerCfg(MyCobotVisualAlignPPORunnerCfg):
    experiment_name = "visual_align_mycobot_280_jn_yolo_hand"


@configclass
class MyCobotYoloHandDomainRandPPORunnerCfg(MyCobotYoloHandPPORunnerCfg):
    max_iterations = 600
    experiment_name = "visual_align_mycobot_280_jn_yolo_hand_dr"
