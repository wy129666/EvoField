import json
import random
import sys

import numpy as np
import torch

from openstl.datasets.dataloader_moving_mnist import MovingMNIST


data_root, validation_path, seeds_path, output_path = sys.argv[1:5]

frozen = np.load(validation_path, mmap_mode="r")
seeds = np.load(seeds_path)

dataset = MovingMNIST(
    root=data_root,
    is_train=True,
    data_name="mnist",
    n_frames_input=10,
    n_frames_output=10,
    image_size=64,
    num_objects=[2],
    use_augment=False,
)


def reset(seed):
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def via_getitem(seed):
    reset(seed)
    inputs, targets = dataset[0]
    sequence = torch.cat([inputs, targets], dim=0)[:, 0]
    return sequence.mul(255).round().clamp(0, 255).byte().numpy()


def via_direct(seed):
    reset(seed)
    sequence = dataset.generate_moving_mnist(
        num_digits=2, background=False
    )
    return np.asarray(sequence[..., 0], dtype=np.uint8)


generators = {
    "dataset_getitem": via_getitem,
    "direct_generate": via_direct,
}

first20 = {}

for name, generator in generators.items():
    matches = 0
    max_diff = 0

    for index in range(20):
        generated = generator(seeds[index])
        reference = np.asarray(frozen[index])

        if np.array_equal(generated, reference):
            matches += 1

        max_diff = max(
            max_diff,
            int(np.abs(
                generated.astype(np.int16)
                - reference.astype(np.int16)
            ).max()),
        )

    first20[name] = {
        "matches": matches,
        "max_abs_diff": max_diff,
    }
    print(name, first20[name])

selected = next(
    (
        name for name, values in first20.items()
        if values["matches"] == 20
    ),
    None,
)

exact = 0
overall_max_diff = 0

if selected is not None:
    generator = generators[selected]

    for index, seed in enumerate(seeds):
        generated = generator(seed)
        reference = np.asarray(frozen[index])

        if np.array_equal(generated, reference):
            exact += 1

        overall_max_diff = max(
            overall_max_diff,
            int(np.abs(
                generated.astype(np.int16)
                - reference.astype(np.int16)
            ).max()),
        )

        if (index + 1) % 200 == 0:
            print(
                f"verified={index + 1}/2000 exact={exact}"
            )

status = (
    "PASS_EXACT_REPRODUCTION"
    if exact == len(seeds)
    else "METADATA_REGEN_MISMATCH"
)

result = {
    "status": status,
    "selected_generator": selected,
    "exact_matches": exact,
    "total_sequences": int(len(seeds)),
    "maximum_absolute_pixel_difference": overall_max_diff,
    "first_20_candidates": first20,
}

with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)

print(json.dumps(result, indent=2))
print("D1 GENERATOR REPRODUCTION:", status)
