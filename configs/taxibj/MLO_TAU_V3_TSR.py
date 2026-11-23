method = 'TAU'
# model
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'tau'
hid_S = 32
hid_T = 256
N_T = 8
N_S = 2
alpha = 0.1
# training
lr = 5e-4
batch_size = 16
drop_path = 0.1
sched = 'cosine'
warmup_epoch = 5

# MLO-TAU V2 overrides copied from the validated MMNIST config
model_type = 'mlo_tau_v3_tsr'
mlo_channels = 80
mlo_groups = 8
mlo_mlp_ratio = 6.0
mlo_init_value = 1e-2

# V3-TSR: nine-way transport plus signed local source residual
mlo_source_hidden_channels = 40
