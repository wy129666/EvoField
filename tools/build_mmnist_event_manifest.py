import csv
import json
import random
import sys
from collections import Counter

import numpy as np
import torch

from openstl.datasets.dataloader_moving_mnist import MovingMNIST


data_root, validation_path, seeds_path, output_dir = sys.argv[1:5]

FUTURE_START = 10
TOTAL_FRAMES = 20
DIGIT_SIZE = 28
IMAGE_SIZE = 64
CANVAS_SIZE = IMAGE_SIZE - DIGIT_SIZE
HISTORY_DV_THRESHOLD = 0.5


class TrackedMovingMNIST(MovingMNIST):
    def tracked_trajectory(self):
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi

        base_vx = np.cos(theta)
        base_vy = np.sin(theta)
        bounce_x = 1
        bounce_y = 1

        top_x = []
        top_y = []
        continuous_x = []
        continuous_y = []
        boundary_hit = []

        for _ in range(TOTAL_FRAMES):
            y += bounce_y * base_vy * self.step_length_
            x += bounce_x * base_vx * self.step_length_

            hit = False

            if x <= 0:
                x = 0
                bounce_x = -bounce_x
                hit = True
            if x >= 1:
                x = 1
                bounce_x = -bounce_x
                hit = True
            if y <= 0:
                y = 0
                bounce_y = -bounce_y
                hit = True
            if y >= 1:
                y = 1
                bounce_y = -bounce_y
                hit = True

            continuous_x.append(x * CANVAS_SIZE)
            continuous_y.append(y * CANVAS_SIZE)
            top_x.append(int(x * CANVAS_SIZE))
            top_y.append(int(y * CANVAS_SIZE))
            boundary_hit.append(hit)

        continuous_x = np.asarray(continuous_x)
        continuous_y = np.asarray(continuous_y)

        velocity_x = np.diff(
            continuous_x, prepend=continuous_x[0]
        )
        velocity_y = np.diff(
            continuous_y, prepend=continuous_y[0]
        )

        return {
            "top_x": np.asarray(top_x, dtype=np.int16),
            "top_y": np.asarray(top_y, dtype=np.int16),
            "center_x": continuous_x + (DIGIT_SIZE - 1) / 2,
            "center_y": continuous_y + (DIGIT_SIZE - 1) / 2,
            "velocity_x": velocity_x,
            "velocity_y": velocity_y,
            "boundary_hit": np.asarray(boundary_hit, dtype=bool),
        }

    def generate_moving_mnist(self, num_digits=2, background=False):
        assert not background

        data = np.zeros(
            (TOTAL_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
            dtype=np.float32,
        )

        trajectories = []
        masks = []

        for digit_index in range(num_digits):
            trajectory = self.tracked_trajectory()

            mnist_id = random.randint(0, self.mnist.shape[0] - 1)
            digit = self.mnist[mnist_id].copy()

            digit_masks = np.zeros(
                (TOTAL_FRAMES, IMAGE_SIZE, IMAGE_SIZE),
                dtype=bool,
            )

            for frame in range(TOTAL_FRAMES):
                top = int(trajectory["top_y"][frame])
                left = int(trajectory["top_x"][frame])
                bottom = top + DIGIT_SIZE
                right = left + DIGIT_SIZE

                data[frame, top:bottom, left:right] = np.maximum(
                    data[frame, top:bottom, left:right],
                    digit,
                )

                digit_masks[
                    frame, top:bottom, left:right
                ] = digit > 0

            trajectory["digit_index"] = digit_index
            trajectory["mnist_id"] = mnist_id

            trajectories.append(trajectory)
            masks.append(digit_masks)

        self.last_trajectories = trajectories
        self.last_masks = np.stack(masks)

        return data[..., np.newaxis]


def reset_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


frozen = np.load(validation_path, mmap_mode="r")
seeds = np.load(seeds_path)

dataset = TrackedMovingMNIST(
    root=data_root,
    is_train=True,
    data_name="mnist",
    n_frames_input=10,
    n_frames_output=10,
    image_size=64,
    num_objects=[2],
    use_augment=False,
)

event_rows = []
horizon_rows = []
trajectory_rows = []

packed_masks = np.empty(
    (len(seeds), 2, TOTAL_FRAMES, IMAGE_SIZE, IMAGE_SIZE // 8),
    dtype=np.uint8,
)

exact_matches = 0

for sequence_id, seed in enumerate(seeds):
    reset_seed(seed)

    inputs, targets = dataset[0]
    regenerated = (
        torch.cat([inputs, targets], dim=0)[:, 0]
        .mul(255)
        .round()
        .byte()
        .numpy()
    )

    if not np.array_equal(regenerated, frozen[sequence_id]):
        raise RuntimeError(
            f"METADATA_REGEN_MISMATCH sequence={sequence_id}"
        )

    exact_matches += 1

    trajectories = dataset.last_trajectories
    masks = dataset.last_masks

    packed_masks[sequence_id] = np.packbits(
        masks, axis=-1
    )

    overlap_area = np.logical_and(
        masks[0], masks[1]
    ).sum(axis=(1, 2))

    digit_areas = masks.sum(axis=(2, 3))
    overlap_denominator = np.minimum(
        digit_areas[0], digit_areas[1]
    ).clip(min=1)

    overlap_ratio = overlap_area / overlap_denominator
    overlap_state = overlap_area > 0

    future_boundary = False
    future_enter = False
    future_exit = False

    for horizon in range(1, 11):
        frame = 9 + horizon
        previous = frame - 1

        boundary = any(
            bool(item["boundary_hit"][frame])
            for item in trajectories
        )
        enter = (
            not bool(overlap_state[previous])
            and bool(overlap_state[frame])
        )
        exit_event = (
            bool(overlap_state[previous])
            and not bool(overlap_state[frame])
        )

        future_boundary |= boundary
        future_enter |= enter
        future_exit |= exit_event

        horizon_rows.append({
            "sequence_id": sequence_id,
            "seed": int(seed),
            "horizon": horizon,
            "frame_index": frame,
            "boundary_turn": int(boundary),
            "overlap_enter": int(enter),
            "overlap_exit": int(exit_event),
            "overlap_area": int(overlap_area[frame]),
            "overlap_ratio": float(overlap_ratio[frame]),
        })

    maximum_history_dv = 0.0
    observed_speed = []

    for item in trajectories:
        vx = item["velocity_x"]
        vy = item["velocity_y"]

        speed = np.sqrt(vx**2 + vy**2)
        delta_v = np.sqrt(
            np.diff(vx)**2 + np.diff(vy)**2
        )

        observed_speed.extend(speed[1:10].tolist())
        maximum_history_dv = max(
            maximum_history_dv,
            float(delta_v[1:9].max(initial=0)),
        )

        for frame in range(TOTAL_FRAMES):
            trajectory_rows.append({
                "sequence_id": sequence_id,
                "seed": int(seed),
                "digit_index": item["digit_index"],
                "mnist_id": item["mnist_id"],
                "frame": frame,
                "center_x": float(item["center_x"][frame]),
                "center_y": float(item["center_y"][frame]),
                "velocity_x": float(item["velocity_x"][frame]),
                "velocity_y": float(item["velocity_y"][frame]),
                "boundary_hit": int(
                    item["boundary_hit"][frame]
                ),
            })

    history_unstable = (
        maximum_history_dv >= HISTORY_DV_THRESHOLD
    )

    future_event_count = sum([
        future_boundary,
        future_enter,
        future_exit,
    ])

    no_future_overlap = not bool(
        overlap_state[9:20].any()
    )

    if future_event_count > 1:
        group = "mixed"
    elif future_boundary:
        group = "boundary_turn"
    elif future_enter:
        group = "overlap_enter"
    elif future_exit:
        group = "overlap_exit"
    elif history_unstable:
        group = "history_unstable"
    elif no_future_overlap:
        group = "simple_constant"
    else:
        group = "mixed"

    minimum_boundary_distance = min(
        float(np.minimum.reduce([
            item["top_x"][10:],
            CANVAS_SIZE - item["top_x"][10:],
            item["top_y"][10:],
            CANVAS_SIZE - item["top_y"][10:],
        ]).min())
        for item in trajectories
    )

    event_rows.append({
        "sequence_id": sequence_id,
        "seed": int(seed),
        "event_group": group,
        "future_boundary_turn": int(future_boundary),
        "future_overlap_enter": int(future_enter),
        "future_overlap_exit": int(future_exit),
        "history_unstable": int(history_unstable),
        "mean_observed_speed": float(np.mean(observed_speed)),
        "maximum_history_delta_v": maximum_history_dv,
        "minimum_future_boundary_distance": minimum_boundary_distance,
        "maximum_future_overlap_ratio": float(
            overlap_ratio[10:20].max()
        ),
        "mean_future_overlap_ratio": float(
            overlap_ratio[10:20].mean()
        ),
    })

    if (sequence_id + 1) % 200 == 0:
        print(
            f"exported={sequence_id + 1}/{len(seeds)}"
        )


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


write_csv(f"{output_dir}/event_manifest.csv", event_rows)
write_csv(
    f"{output_dir}/horizon_event_manifest.csv",
    horizon_rows,
)
write_csv(
    f"{output_dir}/digit_trajectory.csv",
    trajectory_rows,
)

np.savez_compressed(
    f"{output_dir}/per_digit_masks_packed.npz",
    masks=packed_masks,
)

group_counts = Counter(
    row["event_group"] for row in event_rows
)

definition = {
    "status": "PASS_EXACT_TRAJECTORY_METADATA",
    "metadata_type": "exact_generator_trajectory",
    "pixel_exact_matches": exact_matches,
    "total_sequences": int(len(seeds)),
    "history_delta_v_threshold_pixels_per_frame":
        HISTORY_DV_THRESHOLD,
    "prediction_frames": list(range(10, 20)),
    "group_precedence": [
        "mixed_multiple_future_events",
        "boundary_turn",
        "overlap_enter",
        "overlap_exit",
        "history_unstable",
        "simple_constant",
        "mixed_residual",
    ],
    "group_counts": dict(group_counts),
}

with open(
    f"{output_dir}/event_definition.json",
    "w",
    encoding="utf-8",
) as handle:
    json.dump(definition, handle, indent=2)

print(json.dumps(definition, indent=2))
print("D1 EVENT MANIFEST: PASS")
