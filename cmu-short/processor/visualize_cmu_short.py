import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D)

from .data_tools import _some_variables, expmap2rotmat


def set_equal_3d(ax, X, Y, Z, margin=0.2):
    """
    Make 3D axes have equal scale.
    X, Y, Z are 1D arrays of coordinates in *plot* space.
    """
    max_range = max(X.max() - X.min(),
                    Y.max() - Y.min(),
                    Z.max() - Z.min())
    if max_range == 0:
        max_range = 1.0
    mid_x = 0.5 * (X.max() + X.min())
    mid_y = 0.5 * (Y.max() + Y.min())
    mid_z = 0.5 * (Z.max() + Z.min())
    half = max_range * (0.5 + margin)

    ax.set_xlim(mid_x - half, mid_x + half)
    ax.set_ylim(mid_y - half, mid_y + half)
    ax.set_zlim(mid_z - half, mid_z + half)
    ax.set_box_aspect((1, 1, 1))


def channels_to_xyz_seq(channels_seq):
    """
    channels_seq: [T, D] sequence of *denormalized* CMU channels
                  (output_denorm / target_denorm from recognition.py).

    Returns:
        xyz_seq: [T, J, 3]  3D joint positions (J = 38 for CMU skeleton).
    """
    parent, offset, posInd, expmapInd = _some_variables()
    parent = np.asarray(parent, dtype=np.int64)       # [J]
    offset = np.asarray(offset, dtype=np.float32)     # [J, 3]
    njoints = parent.shape[0]

    channels_seq = np.asarray(channels_seq, dtype=np.float32)  # [T, D]
    T, D = channels_seq.shape

    xyz_all = np.zeros((T, njoints, 3), dtype=np.float32)

    for t in range(T):
        angles = channels_seq[t]                      # [D]

        # per-joint rotation & position
        R = [np.eye(3, dtype=np.float32) for _ in range(njoints)]
        xyz = np.zeros((njoints, 3), dtype=np.float32)

        for j in range(njoints):
            r = angles[expmapInd[j]]                 # shape (3,)
            thisR = expmap2rotmat(r)

            if parent[j] == -1:                      # root joint
                R[j] = thisR
                xyz[j] = offset[j]
            else:
                Rp = R[parent[j]]
                xyz[j] = offset[j].dot(Rp) + xyz[parent[j]]
                R[j] = thisR.dot(Rp)

        xyz_all[t] = xyz

    return xyz_all                                   # [T, J, 3]


def animate_stick_3d_from_channels(
        pred_channels,          # [T, D]  (output_denorm)
        gt_channels=None,       # [T, D]  (target_denorm)
        interval=250,           # ms between frames -> slower playback
        save_path=None,
        frame_time_ms=40        # physical time per frame (25 fps -> 40 ms)
):
    """
    3D stickman animation from denormalized CMU channels.

    Conventions:
      - Data uses (x, y, z) with y = up.
      - We plot with:
           X_plot = x          (horizontal)
           Y_plot = z          (depth)
           Z_plot = y          (vertical)
    """

    # ---- channels  3D positions ----
    pred_xyz = channels_to_xyz_seq(pred_channels)      # [T, J, 3]
    if gt_channels is not None:
        gt_xyz = channels_to_xyz_seq(gt_channels)      # [T, J, 3]
        T = min(pred_xyz.shape[0], gt_xyz.shape[0])
        pred_xyz = pred_xyz[:T]
        gt_xyz = gt_xyz[:T]
    else:
        T = pred_xyz.shape[0]

    # Center each frame around root joint (0)
    root_idx = 0
    pred_xyz = pred_xyz - pred_xyz[:, root_idx:root_idx+1, :]
    if gt_channels is not None:
        gt_xyz = gt_xyz - gt_xyz[:, root_idx:root_idx+1, :]

    # Kinematic tree (parents) for edges
    parent, _, _, _ = _some_variables()
    parent = np.asarray(parent, dtype=int)
    njoints = parent.shape[0]

    # Choose a head joint: joint with max Y at first frame (GT preferred)
    if gt_channels is not None:
        ref = gt_xyz[0]
    else:
        ref = pred_xyz[0]
    head_idx = int(np.argmax(ref[:, 1]))   # highest in Y

    # Flatten coords for axis scaling; map data (x,y,z) -> plot (X,Y,Z)
    if gt_channels is not None:
        all_coords = np.concatenate([pred_xyz.reshape(-1, 3),
                                     gt_xyz.reshape(-1, 3)], axis=0)
    else:
        all_coords = pred_xyz.reshape(-1, 3)

    X_all = all_coords[:, 0]   # X_plot = x
    Y_all = all_coords[:, 2]   # Y_plot = z (depth)
    Z_all = all_coords[:, 1]   # Z_plot = y (vertical)

    # Camera: walker passes in front of us
    camera_elev = 15        # slight downward angle
    camera_azim = 0         # looking along -X, walker passes across

    # --------------------------------------------------------
    # Figure + axes: very compact, small gap, titles inside
    # --------------------------------------------------------
    if gt_channels is not None:
        fig = plt.figure(figsize=(6.5, 3.2))
        gs = fig.add_gridspec(
            1, 2,
            left=0.02, right=0.98,
            bottom=0.05, top=0.90,
            wspace=0.02
        )

        ax_gt = fig.add_subplot(gs[0, 0], projection="3d")
        ax_pred = fig.add_subplot(gs[0, 1], projection="3d")

        set_equal_3d(ax_gt, X_all, Y_all, Z_all)
        set_equal_3d(ax_pred, X_all, Y_all, Z_all)

        ax_gt.view_init(elev=camera_elev, azim=camera_azim)
        ax_pred.view_init(elev=camera_elev, azim=camera_azim)

        # Axis labels: X horiz, Z depth, Y vertical
        # ax_gt.set_xlabel("X", labelpad=4, fontsize=8)
        # ax_gt.set_ylabel("Z", labelpad=4, fontsize=8)
        # ax_gt.set_zlabel("Y", labelpad=4, fontsize=8)

        # ax_pred.set_xlabel("X", labelpad=4, fontsize=8)
        # ax_pred.set_ylabel("Z", labelpad=4, fontsize=8)
        # ax_pred.set_zlabel("Y", labelpad=4, fontsize=8)

        # Titles inside each axis (top-left)
        ax_gt.text2D(0.1, 0.97, "Ground Truth", transform=ax_gt.transAxes,
                     ha="left", va="top", fontsize=10, fontweight="bold")
        ax_pred.text2D(0.1, 0.97, "Prediction", transform=ax_pred.transAxes,
                       ha="left", va="top", fontsize=10, fontweight="bold")

        # Bones
        lines_gt = []
        for j in range(njoints):
            if parent[j] == -1:
                continue
            line, = ax_gt.plot([], [], [], lw=3, color="black")
            lines_gt.append((j, parent[j], line))

        lines_pred = []
        for j in range(njoints):
            if parent[j] == -1:
                continue
            line, = ax_pred.plot([], [], [], lw=3, color="black")
            lines_pred.append((j, parent[j], line))

        # Head markers
        head_gt = ax_gt.scatter([], [], [], s=80, marker="o", color="black")
        head_pred = ax_pred.scatter([], [], [], s=80, marker="o", color="black")

        # Per-axis time text (bottom-right inside each plot)
        time_text_gt = ax_gt.text2D(0.9, 0.97, "", transform=ax_gt.transAxes,
                                    ha="right", va="top", fontsize=10, color='blue')
        time_text_pred = ax_pred.text2D(0.9, 0.97, "", transform=ax_pred.transAxes,
                                        ha="right", va="top", fontsize=10, color='blue')
    else:
        fig = plt.figure(figsize=(3.3, 3.2))
        ax_pred = fig.add_subplot(1, 1, 1, projection="3d")

        set_equal_3d(ax_pred, X_all, Y_all, Z_all)
        ax_pred.view_init(elev=camera_elev, azim=camera_azim)

        # ax_pred.set_xlabel("X", labelpad=4, fontsize=8)
        # ax_pred.set_ylabel("Z", labelpad=4, fontsize=8)
        # ax_pred.set_zlabel("Y", labelpad=4, fontsize=8)

        ax_pred.text2D(0.03, 0.97, "Prediction", transform=ax_pred.transAxes,
                       ha="left", va="top", fontsize=10, fontweight="bold")

        lines_gt = None
        lines_pred = []
        for j in range(njoints):
            if parent[j] == -1:
                continue
            line, = ax_pred.plot([], [], [], lw=3, color="black")
            lines_pred.append((j, parent[j], line))

        head_gt = None
        head_pred = ax_pred.scatter([], [], [], s=80, marker="o", color="black")

        time_text_gt = None
        time_text_pred = ax_pred.text2D(0.97, 0.03, "", transform=ax_pred.transAxes,
                                        ha="right", va="top", fontsize=15, color='blue')

    plt.tight_layout()

    # --------------------------------------------------------
    # Animation
    # --------------------------------------------------------
    def init():
        return []

    def animate(t_idx):
        # ---- prediction ----
        joints_p = pred_xyz[t_idx]          # [J,3]  (x,y,z)
        Xp = joints_p[:, 0]                 # x
        Yp = joints_p[:, 2]                 # z (depth)
        Zp = joints_p[:, 1]                 # y (height)

        for j, p, line in lines_pred:
            line.set_data([Xp[p], Xp[j]], [Yp[p], Yp[j]])
            line.set_3d_properties([Zp[p], Zp[j]])

        hx, hy, hz = Xp[head_idx], Yp[head_idx], Zp[head_idx]
        head_pred._offsets3d = ([hx], [hy], [hz])

        # ---- ground truth ----
        if gt_channels is not None:
            joints_g = gt_xyz[t_idx]
            Xg = joints_g[:, 0]
            Yg = joints_g[:, 2]
            Zg = joints_g[:, 1]

            for j, p, line in lines_gt:
                line.set_data([Xg[p], Xg[j]], [Yg[p], Yg[j]])
                line.set_3d_properties([Zg[p], Zg[j]])

            hgx, hgy, hgz = Xg[head_idx], Yg[head_idx], Zg[head_idx]
            head_gt._offsets3d = ([hgx], [hgy], [hgz])

        # ---- timestamps (per axis) ----
        t_ms = t_idx * frame_time_ms
        if time_text_gt is not None:
            time_text_gt.set_text(f"t = {t_ms} ms")
        if time_text_pred is not None:
            time_text_pred.set_text(f"t = {t_ms} ms")

        return []

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=T,
        interval=interval,
        blit=False,
    )

    if save_path is not None:
        if save_path.endswith(".gif"):
            fps = max(1, int(1000 / interval))
            anim.save(save_path, writer="pillow", fps=fps)
        elif save_path.endswith(".mp4"):
            fps = max(1, int(1000 / interval))
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=fps, bitrate=1800)
            anim.save(save_path, writer=writer)

    return anim
