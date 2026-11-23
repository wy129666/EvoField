import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from openstl.models import SimVP_Model
from .base_method import Base_method
from .dcbt_oracle import make_projected_decoder_oracle


def normalized_future_latent_loss(pred_latent, target_latent):
    """Channel-normalized SmoothL1 with an always detached target."""
    if pred_latent.shape != target_latent.shape:
        raise ValueError(
            "Predicted and target latent shapes must match, got "
            f"{tuple(pred_latent.shape)} and {tuple(target_latent.shape)}"
        )
    pred_latent = F.normalize(pred_latent, p=2, dim=2, eps=1e-6)
    target_latent = F.normalize(
        target_latent.detach(), p=2, dim=2, eps=1e-6
    )
    return F.smooth_l1_loss(pred_latent, target_latent)


class SimVP(Base_method):
    r"""SimVP

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, **args):
        super().__init__(**args)
        object.__setattr__(self, '_oracle_teacher', None)
        self._initialize_from_s0(args)

    @staticmethod
    def _checkpoint_sha256(path):
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _extract_model_state(checkpoint):
        raw = checkpoint.get('state_dict', checkpoint)
        state = {
            key[len('model.'):]: value
            for key, value in raw.items()
            if key.startswith('model.')
        }
        return state if state else raw

    def _initialize_from_s0(self, args):
        checkpoint_path = args.get('s0_init_checkpoint')
        dcbt_enabled = bool(args.get('dcbt_enabled', False))
        dcbt_oracle_enabled = bool(
            args.get('dcbt_oracle_enabled', False)
        )

        if dcbt_oracle_enabled and not dcbt_enabled:
            raise ValueError(
                'dcbt_oracle_enabled requires dcbt_enabled=True'
            )

        if dcbt_oracle_enabled and not checkpoint_path:
            raise ValueError(
                'DCBT oracle training requires s0_init_checkpoint'
            )

        if not checkpoint_path:
            return

        expected_sha = str(
            args.get('s0_init_sha256', '')
        ).strip()

        if expected_sha:
            observed_sha = self._checkpoint_sha256(
                checkpoint_path
            )
            if observed_sha != expected_sha:
                raise RuntimeError(
                    'S0 checkpoint SHA256 mismatch: '
                    f'{observed_sha} != {expected_sha}'
                )

        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu',
        )
        model_state = self._extract_model_state(
            checkpoint
        )

        result = self.model.load_state_dict(
            model_state,
            strict=False,
        )

        if result.unexpected_keys:
            raise RuntimeError(
                f'Unexpected S0 keys: '
                f'{result.unexpected_keys}'
            )

        if dcbt_enabled:
            if not result.missing_keys:
                raise RuntimeError(
                    'Expected hid_corr missing keys'
                )
            if not all(
                key.startswith('hid_corr.')
                for key in result.missing_keys
            ):
                raise RuntimeError(
                    f'Invalid missing keys: '
                    f'{result.missing_keys}'
                )
        elif result.missing_keys:
            raise RuntimeError(
                f'Control model missing keys: '
                f'{result.missing_keys}'
            )

        freeze_backbone = bool(
            args.get('freeze_backbone', False)
        )

        if freeze_backbone:
            for module in (
                self.model.enc,
                self.model.hid,
                self.model.dec,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        elif bool(
            args.get('freeze_encoder_decoder', False)
        ):
            for parameter in self.model.enc.parameters():
                parameter.requires_grad_(False)
            for parameter in self.model.dec.parameters():
                parameter.requires_grad_(False)

        if dcbt_oracle_enabled:
            teacher_args = dict(args)
            teacher_args['dcbt_enabled'] = False
            teacher_args['tq_enabled'] = False
            teacher_args['full_tq_enabled'] = False
            teacher_args['pstg_enabled'] = False
            teacher_args['hcmf_enabled'] = False
            teacher_args['mtsc_enabled'] = False

            teacher = SimVP_Model(**teacher_args)
            teacher.load_state_dict(
                model_state,
                strict=True,
            )
            teacher.eval()

            for parameter in teacher.parameters():
                parameter.requires_grad_(False)

            object.__setattr__(
                self,
                '_oracle_teacher',
                teacher,
            )

        del checkpoint
        del model_state

    def _get_oracle_teacher(self, device):
        teacher = object.__getattribute__(
            self,
            '_oracle_teacher',
        )
        if teacher is None:
            raise RuntimeError(
                'DCBT oracle teacher is unavailable'
            )

        teacher_device = next(
            teacher.parameters()
        ).device

        if teacher_device != device:
            teacher.to(device)

        teacher.eval()
        return teacher

    def _enforce_frozen_backbone_eval(self):
        if bool(
            self.hparams.get(
                'freeze_backbone',
                False,
            )
        ):
            self.model.enc.eval()
            self.model.hid.eval()
            self.model.dec.eval()
        elif bool(
            self.hparams.get(
                'freeze_encoder_decoder',
                False,
            )
        ):
            self.model.enc.eval()
            self.model.dec.eval()

    def on_train_epoch_start(self):
        self._enforce_frozen_backbone_eval()

    def _dcbt_training_step(
            self, batch_x, batch_y):
        self._enforce_frozen_backbone_eval()

        teacher = self._get_oracle_teacher(
            batch_x.device
        )

        with torch.no_grad():
            teacher_history, teacher_skip = (
                teacher.encode_with_skip(
                    batch_x.float()
                )
            )
            teacher_nominal = teacher.hid(
                teacher_history
            )

        oracle = make_projected_decoder_oracle(
            teacher=teacher,
            z_nominal=teacher_nominal,
            skip=teacher_skip,
            target=batch_y.float(),
            steps=int(
                self.hparams.get(
                    'oracle_steps',
                    25,
                )
            ),
            budget_fraction=float(
                self.hparams.get(
                    'oracle_budget_fraction',
                    0.01,
                )
            ),
            step_fraction=float(
                self.hparams.get(
                    'oracle_step_fraction',
                    0.25,
                )
            ),
        )

        pred_y, auxiliary = self.model(
            batch_x,
            return_dcbt_aux=True,
        )

        pixel_loss = self.criterion(
            pred_y,
            batch_y,
        )

        scale = (
            teacher_nominal.square()
            .mean(
                dim=(1, 2, 3, 4),
                keepdim=True,
            )
            .sqrt()
            .clamp_min(1e-6)
        )

        predicted_delta = (
            auxiliary['z_hat']
            - teacher_nominal
        ) / scale

        target_delta = (
            oracle['z_star']
            - teacher_nominal
        ) / scale

        oracle_loss = F.smooth_l1_loss(
            predicted_delta,
            target_delta.detach(),
        )

        oracle_weight = float(
            self.hparams.get(
                'lambda_oracle',
                0.1,
            )
        )
        loss = (
            pixel_loss
            + oracle_weight * oracle_loss
        )

        with torch.no_grad():
            recovery = (
                oracle['baseline_errors']
                - oracle['oracle_errors']
            ) / oracle[
                'baseline_errors'
            ].clamp_min(1e-12)

        self.log(
            'train_pixel_loss',
            pixel_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            'train_oracle_loss',
            oracle_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            'train_oracle_recovery',
            recovery.mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            'train_delta_abs',
            auxiliary['delta'].abs().mean(),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def _build_model(self, **args):
        return SimVP_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    
    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch

        if bool(
            self.hparams.get(
                'dcbt_oracle_enabled',
                False,
            )
        ):
            return self._dcbt_training_step(
                batch_x,
                batch_y,
            )

        latent_enabled = bool(self.hparams.get('latent_enabled', False))
        if latent_enabled:
            if self.hparams.aft_seq_length != self.hparams.pre_seq_length:
                raise ValueError(
                    "Initial latent supervision requires equal input and "
                    "output sequence lengths"
                )
            pred_y, pred_latent = self.model(
                batch_x, return_latent=True
            )
            with torch.no_grad():
                target_latent = self.model.encode_frames(batch_y)
            pixel_loss = self.criterion(pred_y, batch_y)
            latent_loss = normalized_future_latent_loss(
                pred_latent, target_latent
            )
            latent_weight = float(
                self.hparams.get('lambda_latent', 0.1)
            )
            if latent_weight < 0.0:
                raise ValueError("lambda_latent must be non-negative")
            loss = pixel_loss + latent_weight * latent_loss
            self.log(
                'train_pixel_loss', pixel_loss,
                on_step=True, on_epoch=True, prog_bar=False
            )
            self.log(
                'train_latent_loss', latent_loss,
                on_step=True, on_epoch=True, prog_bar=False
            )
        else:
            pred_y = self(batch_x)
            loss = self.criterion(pred_y, batch_y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
