import torch

from openstl.models import PredNet_Model
from openstl.utils import get_initial_states
from .base_method import Base_method


class PredNet(Base_method):
    r"""Lightning adapter for the historical OpenSTL PredNet core.

    The model architecture, autoregressive extrapolation, and L0
    prediction-error training objective follow the historical OpenSTL
    PredNet implementation.
    """

    def _build_model(self, **args):
        return PredNet_Model(
            self.hparams.stack_sizes,
            self.hparams.R_stack_sizes,
            self.hparams.A_filt_sizes,
            self.hparams.Ahat_filt_sizes,
            self.hparams.R_filt_sizes,
            self.hparams.pixel_max,
            self.hparams,
        )

    def _initial_states(self, sequence):
        return get_initial_states(
            sequence.shape,
            -2,
            -1,
            len(self.hparams.stack_sizes),
            self.hparams.R_stack_sizes,
            self.hparams.stack_sizes,
            -3,
            sequence.device,
        )

    def _prediction_error_loss(self, error_list):
        num_layers = len(self.hparams.stack_sizes)

        if self.hparams.weight_mode == "L_0":
            layer_weights = error_list[0].new_zeros(num_layers)
            layer_weights[0] = 1.0
        elif self.hparams.weight_mode == "L_all":
            layer_weights = error_list[0].new_full(
                (num_layers,), 0.1
            )
            layer_weights[0] = 1.0
        else:
            raise RuntimeError(
                "weight_mode must be L_0 or L_all"
            )

        time_denominator = len(error_list) - 1
        total_error = error_list[0].sum() * 0.0

        for error_at_t in error_list[1:]:
            total_error = total_error + torch.sum(
                error_at_t * layer_weights
            ) / time_denominator

        return total_error / error_list[0].shape[0]

    def forward(self, batch_x, batch_y, **kwargs):
        sequence = torch.cat([batch_x, batch_y], dim=1)
        states = self._initial_states(sequence)

        predictions, _ = self.model(
            sequence,
            states,
            extrapolation=True,
        )

        return torch.stack(
            predictions[batch_x.shape[1]:],
            dim=1,
        )

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        sequence = torch.cat([batch_x, batch_y], dim=1)
        states = self._initial_states(sequence)

        _, error_list = self.model(
            sequence,
            states,
            extrapolation=False,
        )

        loss = self._prediction_error_loss(error_list)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss
