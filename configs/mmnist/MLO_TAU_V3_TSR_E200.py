method = 'TAU'

spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'mlo_tau_v3_tsr'

hid_S = 64
hid_T = 512
N_T = 8
N_S = 4

# Official TAU DDR coefficient.
alpha = 0.1

mlo_channels = 80
mlo_groups = 8
mlo_mlp_ratio = 6.0
mlo_init_value = 1e-2

dcbt_enabled = False
tq_enabled = False
full_tq_enabled = False
pstg_enabled = False
hcmf_enabled = False
lr = 1e-3
batch_size = 16
drop = 0.0
drop_path = 0.0
sched = 'cosine'

in_shape = [10, 1, 64, 64]

pre_seq_length = 10

aft_seq_length = 10

epoch = 200

warmup_epoch = 0

val_batch_size = 16

accumulate_grad_batches = 1

mlo_source_hidden_channels = 40
