import torch
import torch.nn.functional as F


def per_sample_mse(prediction, target):
    return (
        prediction.float() - target.float()
    ).square().flatten(1).mean(dim=1)


@torch.enable_grad()
def make_projected_decoder_oracle(
        teacher,
        z_nominal,
        skip,
        target,
        steps=25,
        budget_fraction=0.01,
        step_fraction=0.25):

    z_nominal = z_nominal.detach().float()
    skip = skip.detach().float()
    target = target.detach().float()

    with torch.no_grad():
        baseline_prediction = teacher.decode_latent(
            z_nominal,
            skip,
        )
        baseline_errors = per_sample_mse(
            baseline_prediction,
            target,
        )

    batch_size = z_nominal.shape[0]
    reference_norm = (
        z_nominal.flatten(1)
        .norm(dim=1)
        .clamp_min(1e-12)
    )
    budget = budget_fraction * reference_norm

    delta = torch.zeros_like(
        z_nominal,
        requires_grad=True,
    )

    best_errors = baseline_errors.clone()
    best_delta = torch.zeros_like(z_nominal)
    best_steps = torch.zeros(
        batch_size,
        dtype=torch.int64,
        device=z_nominal.device,
    )

    for step in range(steps):
        prediction = teacher.decode_latent(
            z_nominal + delta,
            skip,
        )
        errors = per_sample_mse(prediction, target)

        with torch.no_grad():
            improved = errors < best_errors
            best_errors[improved] = errors[improved]
            best_delta[improved] = delta.detach()[improved]
            best_steps[improved] = step

        loss = errors.mean()
        gradient, = torch.autograd.grad(
            loss,
            delta,
            create_graph=False,
            retain_graph=False,
        )

        with torch.no_grad():
            flat_delta = delta.flatten(1)
            flat_gradient = gradient.flatten(1)

            gradient_norm = (
                flat_gradient.norm(dim=1, keepdim=True)
                .clamp_min(1e-12)
            )

            flat_delta.sub_(
                step_fraction
                * budget[:, None]
                * flat_gradient
                / gradient_norm
            )

            delta_norm = (
                flat_delta.norm(dim=1, keepdim=True)
                .clamp_min(1e-12)
            )
            projection = torch.clamp(
                budget[:, None] / delta_norm,
                max=1.0,
            )
            flat_delta.mul_(projection)

        delta = delta.detach().requires_grad_(True)

    with torch.no_grad():
        final_prediction = teacher.decode_latent(
            z_nominal + delta,
            skip,
        )
        final_errors = per_sample_mse(
            final_prediction,
            target,
        )
        improved = final_errors < best_errors
        best_errors[improved] = final_errors[improved]
        best_delta[improved] = delta.detach()[improved]
        best_steps[improved] = steps

        delta_ratio = (
            best_delta.flatten(1).norm(dim=1)
            / reference_norm
        )

    return {
        "z_star": z_nominal + best_delta,
        "delta_star": best_delta,
        "baseline_errors": baseline_errors,
        "oracle_errors": best_errors,
        "delta_ratio": delta_ratio,
        "best_steps": best_steps,
    }
