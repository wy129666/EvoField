method = 'TAU'

# Human3.6M-Walking 128x128 unified local protocol
in_shape = [4, 3, 128, 128]
pre_seq_length = 4
aft_seq_length = 4

# Matched TAU backbone
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'mlo_tau_v3_tsr'
hid_S = 64
hid_T = 512
N_T = 8
N_S = 4
alpha = 0.1

# MLO-TAU V3-TSR
mlo_channels = 80
mlo_groups = 8
mlo_mlp_ratio = 6.0
mlo_init_value = 1e-2
mlo_source_hidden_channels = 40

# Matched training protocol
lr = 1e-3
batch_size = 2
val_batch_size = 2
accumulate_grad_batches = 8
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 0

epoch = 100
