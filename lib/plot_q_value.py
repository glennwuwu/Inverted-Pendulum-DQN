"""Plot Q-Values for all States"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_state_q(
    q_model, action_space, episode=0, save_name="img/dqn", save_image=False
):
    points_per_ax = 500
    half_points_per_ax = int(points_per_ax / 2)

    theta_1 = np.linspace(np.pi, 0, half_points_per_ax)
    theta_2 = np.linspace(2 * np.pi, np.pi, half_points_per_ax)
    theta_ = np.concatenate((theta_1, theta_2))
    x_ = np.cos(theta_)
    y_ = np.sin(theta_)
    dtheta_dt_ = np.linspace(-8, 8, points_per_ax)

    states_ = [(x, y, dtheta_dt) for dtheta_dt in dtheta_dt_ for x, y in zip(x_, y_)]

    states_ = torch.tensor(states_, dtype=torch.float32).to("cuda")
    with torch.no_grad():
        q_ = q_model(states_).cpu().numpy()
    max_q = np.max(q_, axis=1).reshape((points_per_ax, points_per_ax))
    # min_q = np.min(q_, axis=1).reshape((points_per_ax, points_per_ax))
    action = np.argmax(q_, axis=1)
    torque = np.array([action_space[a] for a in action]).reshape(
        (points_per_ax, points_per_ax)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))  # 1 row, 2 columns

    # Plot for Maximum Q-Value
    im1 = axes[0].imshow(
        max_q,
        extent=(0, 1, min(dtheta_dt_), max(dtheta_dt_)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[0].set_title(f"Maximum Q-Value (Episode {episode})")
    axes[0].set_xlabel("Theta")
    axes[0].set_ylabel("Angular Velocity")
    axes[0].set_xticks(
        [0, 0.25, 0.5, 0.75, 1], ["pi", "1/2pi", "0 or 2pi", "3/2pi", "pi"]
    )
    axes[0].grid(False)

    # Plot for Action
    im2 = axes[1].imshow(
        torque,
        extent=(0, 1, min(dtheta_dt_), max(dtheta_dt_)),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    axes[1].set_title(f"Torque (Episode {episode})")
    axes[1].set_xlabel("Theta")
    axes[1].set_ylabel("Angular Velocity")
    axes[1].set_xticks(
        [0, 0.25, 0.5, 0.75, 1], ["pi", "1/2pi", "0 or 2pi", "3/2pi", "pi"]
    )
    axes[1].grid(False)

    #     plt.subplots_adjust(bottom=0)

    cbar1 = fig.colorbar(im1, ax=axes[0], orientation="vertical")
    cbar2 = fig.colorbar(im2, ax=axes[1], orientation="vertical")

    plt.tight_layout()  # Adjust layout to prevent overlapping

    if save_image:
        plt.savefig(f"{save_name}-{episode:04d}.png")
        plt.close()
    else:
        plt.show()
