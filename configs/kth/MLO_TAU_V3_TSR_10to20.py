method = 'TAU'

# KTH 10 -> 20, PFGNet/OpenSTL table-compatible protocol
spatio_kernel_enc = 3
spatio_kernel_dec = 3
model_type = 'mlo_tau_v3_tsr'

hid_S = 64
hid_T = 256
N_T = 6
N_S = 2
alpha = 0.1

mlo_channels = 80
mlo_groups = 8
mlo_mlp_ratio = 6.0
mlo_init_value = 1e-2
mlo_source_hidden_channels = 40

pre_seq_length = 10
aft_seq_length = 20
in_shape = [10, 1, 128, 128]
kth_protocol = 'pfg_official'

lr = 2e-4
batch_size = 2
val_batch_size = 2
accumulate_grad_batches = 4
drop_path = 0.1
sched = 'onecycle'
