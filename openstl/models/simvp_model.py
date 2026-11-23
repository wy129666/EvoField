import torch
from torch import nn

from openstl.modules.dcbt_modules import DCBTCorrection

from openstl.modules.pstg_modules import PredictiveSpatioTemporalGate

from openstl.modules.hcmf_modules import HorizonConditionedInception

from openstl.modules import (ConvSC, ConvNeXtSubBlock, ConvMixerSubBlock, GASubBlock, gInception_ST,
                             HorNetSubBlock, MLPMixerSubBlock, MogaSubBlock, PoolFormerSubBlock,
                             SwinSubBlock, UniformerSubBlock, VANSubBlock, ViTSubBlock, TAUSubBlock)
from openstl.modules.temporal_query_adapter import TemporalQueryAdapter
from openstl.modules.explicit_time_memory import ExplicitTimeMemory
from openstl.modules.future_query_decoder import FullTemporalQuery


from openstl.modules.mlo_tau_modules import MLOTAUTranslator
from openstl.modules.mlo_tau_v2_modules import MLOTAUV2Translator
from openstl.modules.mlo_tau_v3_tsr_modules import MLOTAUV3TSRTranslator

class SimVP_Model(nn.Module):
    r"""SimVP Model

    Implementation of `SimVP: Simpler yet Better Video Prediction
    <https://arxiv.org/abs/2206.05099>`_.

    """

    def __init__(self, in_shape, hid_S=16, hid_T=256, N_S=4, N_T=4, model_type='gSTA',
                 mlp_ratio=8., drop=0.0, drop_path=0.0, spatio_kernel_enc=3,
                 spatio_kernel_dec=3, act_inplace=True, tq_enabled=False,
                 tq_heads=4, tq_layers=1, tq_scale_init=0.0,
                 tq_use_time=True, tq_max_future_steps=20,
                 timestamp_mode='relative_time', full_tq_enabled=False,
                 full_tq_use_time=False, full_tq_heads=4, full_tq_layers=2,
                 full_tq_context_width=7168, full_tq_max_future_steps=20,
                 full_tq_skip_mode='original',
                 hcmf_enabled=False, hcmf_hidden=16,
                 pstg_enabled=False,
                 pstg_use_spatial_gate=True,
                 pstg_use_temporal_gate=True,
                 pstg_use_ddr_loss=False,
                 pstg_scale_init=1e-2, **kwargs):
        super(SimVP_Model, self).__init__()
        T, C, H, W = in_shape  # T is pre_seq_length
        H, W = int(H / 2**(N_S/2)), int(W / 2**(N_S/2))  # downsample 1 / 2**(N_S/2)
        act_inplace = False
        self.enc = Encoder(C, hid_S, N_S, spatio_kernel_enc, act_inplace=act_inplace)
        self.dec = Decoder(hid_S, C, N_S, spatio_kernel_dec, act_inplace=act_inplace)

        self.pstg_enabled = bool(pstg_enabled)

        if pstg_use_ddr_loss:
            raise ValueError(
                "DDR loss is disabled during PSTG structure screening."
            )

        if self.pstg_enabled:
            self.pstg = PredictiveSpatioTemporalGate(
                channels=hid_S,
                use_spatial_gate=pstg_use_spatial_gate,
                use_temporal_gate=pstg_use_temporal_gate,
                scale_init=pstg_scale_init,
            )
        else:
            self.pstg = nn.Identity()

        self.tq_enabled = bool(tq_enabled)
        self.full_tq_enabled = bool(full_tq_enabled)
        if self.tq_enabled and self.full_tq_enabled:
            raise ValueError("tq_enabled and full_tq_enabled are mutually exclusive")
        self.timestamp_mode = timestamp_mode
        if self.timestamp_mode != 'relative_time':
            raise ValueError(
                f"Unsupported timestamp_mode: {self.timestamp_mode}"
            )
        if self.tq_enabled:
            self.tq_adapter = TemporalQueryAdapter(
                channels=hid_S,
                num_heads=tq_heads,
                num_layers=tq_layers,
                scale_init=tq_scale_init,
                use_time=tq_use_time,
                max_future_steps=tq_max_future_steps,
            )

        self.full_tq_skip_mode = full_tq_skip_mode
        if self.full_tq_skip_mode not in {'original', 'last', 'none'}:
            raise ValueError(
                "full_tq_skip_mode must be 'original', 'last', or 'none'"
            )
        if self.full_tq_enabled:
            memory_encoder = ExplicitTimeMemory(
                channels=hid_S,
                num_heads=full_tq_heads,
                num_layers=full_tq_layers,
                context_width=full_tq_context_width,
            )
            self.full_tq = FullTemporalQuery(
                channels=hid_S,
                memory_encoder=memory_encoder,
                num_heads=full_tq_heads,
                num_layers=full_tq_layers,
                max_future_steps=full_tq_max_future_steps,
                use_time=full_tq_use_time,
            )

        if not self.full_tq_enabled:
            model_type = 'gsta' if model_type is None else model_type.lower()
            if model_type == 'mlo_tau_v3_tsr':
                self.hid = MLOTAUV3TSRTranslator(
                    time_steps=T,
                    in_channels=hid_S,
                    channels=int(
                        kwargs.get('mlo_channels', 80)
                    ),
                    depth=N_T,
                    groups=int(
                        kwargs.get('mlo_groups', 8)
                    ),
                    mlp_ratio=float(
                        kwargs.get('mlo_mlp_ratio', 6.0)
                    ),
                    drop=drop,
                    drop_path=drop_path,
                    init_value=float(
                        kwargs.get('mlo_init_value', 1e-2)
                    ),
                    source_hidden_channels=kwargs.get(
                        'mlo_source_hidden_channels', None
                    ),
                )
            elif model_type == 'mlo_tau_v2':
                self.hid = MLOTAUV2Translator(
                    time_steps=T,
                    in_channels=hid_S,
                    channels=int(
                        kwargs.get('mlo_channels', 80)
                    ),
                    depth=N_T,
                    groups=int(
                        kwargs.get('mlo_groups', 8)
                    ),
                    mlp_ratio=float(
                        kwargs.get('mlo_mlp_ratio', 6.0)
                    ),
                    drop=drop,
                    drop_path=drop_path,
                    init_value=float(
                        kwargs.get('mlo_init_value', 1e-2)
                    ),
                )
            elif model_type == 'mlo_tau':
                basis_specs = kwargs.get(
                    'mlo_basis_specs',
                    ((3, 3), (3, 7), (5, 5), (5, 11)),
                )
                basis_specs = tuple(
                    tuple(int(value) for value in pair)
                    for pair in basis_specs
                )
                self.hid = MLOTAUTranslator(
                    time_steps=T,
                    in_channels=hid_S,
                    channels=int(
                        kwargs.get('mlo_channels', 80)
                    ),
                    depth=N_T,
                    groups=int(
                        kwargs.get('mlo_groups', 8)
                    ),
                    basis_specs=basis_specs,
                    mlp_ratio=float(
                        kwargs.get('mlo_mlp_ratio', 6.0)
                    ),
                    drop=drop,
                    drop_path=drop_path,
                    init_value=float(
                        kwargs.get('mlo_init_value', 1e-2)
                    ),
                )
            elif model_type == 'incepu':
                self.hid = MidIncepNet(
                    T*hid_S,
                    hid_T,
                    N_T,
                    hcmf_enabled=hcmf_enabled,
                    num_horizons=T,
                    hcmf_hidden=hcmf_hidden,
                )
            else:
                self.hid = MidMetaNet(T*hid_S, hid_T, N_T,
                    input_resolution=(H, W), model_type=model_type,
                    mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)

        self.dcbt_enabled = bool(
            kwargs.get('dcbt_enabled', False)
        )

        if self.dcbt_enabled:
            if str(model_type).lower() not in ('incepu', 'tau'):
                raise ValueError(
                    "DCBT is restricted to the IncepU or TAU Translator"
                )
            if (
                self.tq_enabled
                or self.full_tq_enabled
                or self.pstg_enabled
                or bool(hcmf_enabled)
            ):
                raise ValueError(
                    "DCBT cannot be combined with TQ, Full-TQ, "
                    "PSTG or HCMF during screening"
                )

            self.hid_corr = DCBTCorrection(
                time_steps=T,
                channels=hid_S,
                hid_t=hid_T,
                n_blocks=int(
                    kwargs.get('dcbt_blocks', N_T)
                ),
                width_ratio=float(
                    kwargs.get('dcbt_width_ratio', 0.5)
                ),
                kernels=tuple(
                    kwargs.get(
                        'dcbt_kernels',
                        (3, 7, 11),
                    )
                ),
                max_delta_ratio=kwargs.get(
                    'dcbt_max_delta_ratio',
                    None,
                ),
            )
        else:
            self.hid_corr = None

    def encode_with_skip(self, frames):
        batch, steps, channels, height, width = frames.shape
        encoded, skip = self.enc(
            frames.reshape(
                batch * steps,
                channels,
                height,
                width,
            )
        )
        _, latent_channels, latent_h, latent_w = encoded.shape
        latent = encoded.reshape(
            batch,
            steps,
            latent_channels,
            latent_h,
            latent_w,
        )
        return latent, skip

    def decode_latent(self, latent, skip):
        batch, steps, channels, height, width = latent.shape
        decoded = self.dec(
            latent.reshape(
                batch * steps,
                channels,
                height,
                width,
            ),
            skip,
        )
        return decoded.reshape(
            batch,
            steps,
            *decoded.shape[1:],
        )


    def _prepare_full_tq_skip(self, skip, batch_size, history_length,
                              future_length):
        if self.full_tq_skip_mode == 'original':
            if future_length != history_length:
                raise ValueError(
                    "original skip requires future_length == history_length; "
                    "use full_tq_skip_mode='last' or 'none' for cross-horizon runs"
                )
            return skip

        _, channels, height, width = skip.shape
        skip = skip.reshape(
            batch_size, history_length, channels, height, width
        )
        if self.full_tq_skip_mode == 'last':
            skip = skip[:, -1:].expand(
                batch_size, future_length, channels, height, width
            )
            return skip.reshape(
                batch_size * future_length, channels, height, width
            )
        return skip.new_zeros(
            batch_size * future_length, channels, height, width
        )

    def encode_frames(self, frames):
        """Encode frames independently on the shared spatial path."""
        B, T, C, H, W = frames.shape
        encoded, _ = self.enc(frames.reshape(B*T, C, H, W))
        _, C_, H_, W_ = encoded.shape
        return encoded.reshape(B, T, C_, H_, W_)

    def forward(self, x_raw, return_latent=False, return_dcbt_aux=False, **kwargs):
        B, T, C, H, W = x_raw.shape
        z, skip = self.encode_with_skip(x_raw)
        _, _, C_, H_, W_ = z.shape

        if self.pstg_enabled:
            z = self.pstg(z)
        z_nominal = None
        delta = None

        if self.full_tq_enabled:
            hid = self.full_tq(
                history=z,
                future_length=kwargs.get('future_length'),
                t_hist=kwargs.get('t_hist'),
                t_future=kwargs.get('t_future'),
                history_mask=kwargs.get('history_mask'),
                slot_indices=kwargs.get('slot_indices'),
            )
            future_length = hid.shape[1]
            skip = self._prepare_full_tq_skip(
                skip, B, T, future_length
            )
        else:
            z_nominal = self.hid(z)
            hid = z_nominal

            if self.dcbt_enabled:
                delta = self.hid_corr(z, z_nominal)
                hid = z_nominal + delta

            if self.tq_enabled:
                hid = self.tq_adapter(
                    z_base=hid,
                    e_hist=z,
                    t_hist=kwargs.get('t_hist'),
                    t_future=kwargs.get('t_future'),
                    history_mask=kwargs.get('history_mask'),
                )
            future_length = T

        latent_output = hid
        Y = self.decode_latent(latent_output, skip)

        if return_dcbt_aux:
            if not self.dcbt_enabled:
                raise ValueError(
                    "return_dcbt_aux requires dcbt_enabled=True"
                )
            return Y, {
                'z_history': z,
                'z_nominal': z_nominal,
                'delta': delta,
                'z_hat': latent_output,
            }

        if return_latent:
            return Y, latent_output
        return Y


def sampling_generator(N, reverse=False):
    samplings = [False, True] * (N // 2)
    if reverse: return list(reversed(samplings[:N]))
    else: return samplings[:N]


class Encoder(nn.Module):
    """3D Encoder for SimVP"""

    def __init__(self, C_in, C_hid, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S)
        super(Encoder, self).__init__()
        self.enc = nn.Sequential(
              ConvSC(C_in, C_hid, spatio_kernel, downsampling=samplings[0],
                     act_inplace=act_inplace),
            *[ConvSC(C_hid, C_hid, spatio_kernel, downsampling=s,
                     act_inplace=act_inplace) for s in samplings[1:]]
        )

    def forward(self, x):  # B*4, 3, 128, 128
        enc1 = self.enc[0](x)
        latent = enc1
        for i in range(1, len(self.enc)):
            latent = self.enc[i](latent)
        return latent, enc1


class Decoder(nn.Module):
    """3D Decoder for SimVP"""

    def __init__(self, C_hid, C_out, N_S, spatio_kernel, act_inplace=True):
        samplings = sampling_generator(N_S, reverse=True)
        super(Decoder, self).__init__()
        self.dec = nn.Sequential(
            *[ConvSC(C_hid, C_hid, spatio_kernel, upsampling=s,
                     act_inplace=act_inplace) for s in samplings[:-1]],
              ConvSC(C_hid, C_hid, spatio_kernel, upsampling=samplings[-1],
                     act_inplace=act_inplace)
        )
        self.readout = nn.Conv2d(C_hid, C_out, 1)

    def forward(self, hid, enc1=None):
        for i in range(0, len(self.dec)-1):
            hid = self.dec[i](hid)
        Y = self.dec[-1](hid + enc1)
        Y = self.readout(Y)
        return Y


class MidIncepNet(nn.Module):
    """The hidden Translator of IncepNet for SimVPv1"""

    def __init__(
            self, channel_in, channel_hid, N2,
            incep_ker=[3,5,7,11], groups=8,
            hcmf_enabled=False, num_horizons=None,
            hcmf_hidden=16, **kwargs):
        super(MidIncepNet, self).__init__()
        assert N2 >= 2 and len(incep_ker) > 1
        self.N2 = N2
        enc_layers = [gInception_ST(
            channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1,N2-1):
            enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        enc_layers.append(
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        dec_layers = [
                gInception_ST(channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups)]
        for i in range(1,N2-1):
            dec_layers.append(
                gInception_ST(2*channel_hid, channel_hid//2, channel_hid,
                              incep_ker=incep_ker, groups=groups))
        if hcmf_enabled:
            if num_horizons is None:
                raise ValueError(
                    "num_horizons is required when HCMF is enabled"
                )
            dec_layers.append(
                HorizonConditionedInception(
                    2*channel_hid,
                    channel_hid//2,
                    channel_in,
                    num_horizons=num_horizons,
                    hidden_dim=hcmf_hidden,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )
        else:
            dec_layers.append(
                gInception_ST(
                    2*channel_hid,
                    channel_hid//2,
                    channel_in,
                    incep_ker=incep_ker,
                    groups=groups,
                )
            )

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        # encoder
        skips = []
        z = x
        for i in range(self.N2):
            z = self.enc[i](z)
            if i < self.N2-1:
                skips.append(z)
        # decoder
        z = self.dec[0](z)
        for i in range(1,self.N2):
            z = self.dec[i](torch.cat([z, skips[-i]], dim=1) )

        y = z.reshape(B, T, C, H, W)
        return y


class MetaBlock(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(MetaBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'hornet':
            self.block = HorNetSubBlock(in_channels, mlp_ratio=mlp_ratio, drop_path=drop_path)
        elif model_type in ['mlp', 'mlpmixer']:
            self.block = MLPMixerSubBlock(
                in_channels, input_resolution, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type in ['moga', 'moganet']:
            self.block = MogaSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop_rate=drop, drop_path_rate=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'swin':
            self.block = SwinSubBlock(
                in_channels, input_resolution, layer_i=layer_i, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'van':
            self.block = VANSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'tau':
            self.block = TAUSubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        else:
            assert False and "Invalid model_type in SimVP"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)


class MidMetaNet(nn.Module):
    """The hidden Translator of MetaFormer for SimVP"""

    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(MidMetaNet, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [  # stochastic depth decay rule
            x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # downsample
        enc_layers = [MetaBlock(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]
        # middle layers
        for i in range(1, N2-1):
            enc_layers.append(MetaBlock(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))
        # upsample
        enc_layers.append(MetaBlock(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T*C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y
