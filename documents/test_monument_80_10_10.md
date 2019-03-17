petrichor@DragonFeaster:~/Projects/SOC_Dbpedia/nmt$ python -m nmt.nmt \
>     --src=en --tgt=sparql \
>     --vocab_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/vocab  \
>     --train_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/train \
>     --dev_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/dev  \
>     --test_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/train \
>     --out_dir=/tmp/nmt_model \
>     --num_train_steps=12000 \
>     --steps_per_stats=100 \
>     --num_layers=2 \
>     --num_units=128 \
>     --dropout=0.2 \
>     --metrics=bleu
# Job id 0
2019-03-09 01:58:30.267407: I tensorflow/stream_executor/cuda/cuda_gpu_executor.cc:964] successful NUMA node read from SysFS had negative value (-1), but there must be at least one NUMA node, so returning NUMA node zero
2019-03-09 01:58:30.267862: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1406] Found device 0 with properties: 
name: GeForce GTX 1050 major: 6 minor: 1 memoryClockRate(GHz): 1.493
pciBusID: 0000:01:00.0
totalMemory: 3.95GiB freeMemory: 3.33GiB
2019-03-09 01:58:30.267882: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1485] Adding visible gpu devices: 0
2019-03-09 01:58:30.563355: I tensorflow/core/common_runtime/gpu/gpu_device.cc:966] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 01:58:30.563403: I tensorflow/core/common_runtime/gpu/gpu_device.cc:972]      0 
2019-03-09 01:58:30.563413: I tensorflow/core/common_runtime/gpu/gpu_device.cc:985] 0:   N 
2019-03-09 01:58:30.563763: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1098] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3045 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
# Devices visible to TensorFlow: [_DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 268435456, 18418720448931006341), _DeviceAttributes(/job:localhost/replica:0/task:0/device:GPU:0, GPU, 3193307136, 8423164175289441783)]
# Creating output directory /tmp/nmt_model ...
# Vocab file /home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/vocab.en exists
The first 3 vocab words [trenton, brevet, selenography] are not [<unk>, <s>, </s>]
# Vocab file /home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/vocab.sparql exists
The first 3 vocab words [dbr_Cassava), var_d, dbr_Timeline_of_Mars_Science_Laboratory] are not [<unk>, <s>, </s>]
  saving hparams to /tmp/nmt_model/hparams
  saving hparams to /tmp/nmt_model/best_bleu/hparams
  attention=
  attention_architecture=standard
  avg_ckpts=False
  batch_size=128
  beam_width=0
  best_bleu=0
  best_bleu_dir=/tmp/nmt_model/best_bleu
  check_special_token=True
  colocate_gradients_with_ops=True
  decay_scheme=
  dev_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/dev
  dropout=0.2
  embed_prefix=None
  encoder_type=uni
  eos=</s>
  epoch_step=0
  forget_bias=1.0
  infer_batch_size=32
  infer_mode=greedy
  init_op=uniform
  init_weight=0.1
  language_model=False
  learning_rate=1.0
  length_penalty_weight=0.0
  log_device_placement=False
  max_gradient_norm=5.0
  max_train=0
  metrics=['bleu']
  num_buckets=5
  num_dec_emb_partitions=0
  num_decoder_layers=2
  num_decoder_residual_layers=0
  num_embeddings_partitions=0
  num_enc_emb_partitions=0
  num_encoder_layers=2
  num_encoder_residual_layers=0
  num_gpus=1
  num_inter_threads=0
  num_intra_threads=0
  num_keep_ckpts=5
  num_sampled_softmax=0
  num_train_steps=12000
  num_translations_per_input=1
  num_units=128
  optimizer=sgd
  out_dir=/tmp/nmt_model
  output_attention=True
  override_loaded_hparams=False
  pass_hidden_state=True
  random_seed=None
  residual=False
  sampling_temperature=0.0
  share_vocab=False
  sos=<s>
  src=en
  src_embed_file=
  src_max_len=50
  src_max_len_infer=None
  src_vocab_file=/tmp/nmt_model/vocab.en
  src_vocab_size=2526
  steps_per_external_eval=None
  steps_per_stats=100
  subword_option=
  test_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/train
  tgt=sparql
  tgt_embed_file=
  tgt_max_len=50
  tgt_max_len_infer=None
  tgt_vocab_file=/tmp/nmt_model/vocab.sparql
  tgt_vocab_size=2530
  time_major=True
  train_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/train
  unit_type=lstm
  use_char_encode=False
  vocab_prefix=/home/petrichor/Projects/SOC_Dbpedia/New_start_March/NSpM/data/monument_300/vocab
  warmup_scheme=t2t
  warmup_steps=0
# Creating train graph ...
# Build a basic encoder
  num_layers = 2, num_residual_layers=0
  cell 0  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
  cell 0  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DropoutWrapper, dropout=0.2   DeviceWrapper, device=/gpu:0
  learning_rate=1, warmup_steps=0, warmup_scheme=t2t
  decay_scheme=, start_decay_step=12000, decay_steps 0, decay_factor 1
# Trainable variables
Format: <name>, <shape>, <(soft) device placement>
  embeddings/encoder/embedding_encoder:0, (2526, 128), /device:GPU:0
  embeddings/decoder/embedding_decoder:0, (2530, 128), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 2530), /device:GPU:0
# Creating eval graph ...
# Build a basic encoder
  num_layers = 2, num_residual_layers=0
  cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
# Trainable variables
Format: <name>, <shape>, <(soft) device placement>
  embeddings/encoder/embedding_encoder:0, (2526, 128), /device:GPU:0
  embeddings/decoder/embedding_decoder:0, (2530, 128), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 2530), /device:GPU:0
# Creating infer graph ...
# Build a basic encoder
  num_layers = 2, num_residual_layers=0
  cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 0  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  cell 1  LSTM, forget_bias=1  DeviceWrapper, device=/gpu:0
  decoder: infer_mode=greedybeam_width=0, length_penalty=0.000000
# Trainable variables
Format: <name>, <shape>, <(soft) device placement>
  embeddings/encoder/embedding_encoder:0, (2526, 128), /device:GPU:0
  embeddings/decoder/embedding_decoder:0, (2530, 128), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/encoder/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_0/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/kernel:0, (256, 512), /device:GPU:0
  dynamic_seq2seq/decoder/multi_rnn_cell/cell_1/basic_lstm_cell/bias:0, (512,), /device:GPU:0
  dynamic_seq2seq/decoder/output_projection/kernel:0, (128, 2530), 
# log_file=/tmp/nmt_model/log_1552076912
2019-03-09 01:58:32.566636: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1485] Adding visible gpu devices: 0
2019-03-09 01:58:32.566683: I tensorflow/core/common_runtime/gpu/gpu_device.cc:966] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 01:58:32.566709: I tensorflow/core/common_runtime/gpu/gpu_device.cc:972]      0 
2019-03-09 01:58:32.566714: I tensorflow/core/common_runtime/gpu/gpu_device.cc:985] 0:   N 
2019-03-09 01:58:32.566869: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1098] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3045 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-03-09 01:58:32.567175: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1485] Adding visible gpu devices: 0
2019-03-09 01:58:32.567207: I tensorflow/core/common_runtime/gpu/gpu_device.cc:966] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 01:58:32.567214: I tensorflow/core/common_runtime/gpu/gpu_device.cc:972]      0 
2019-03-09 01:58:32.567238: I tensorflow/core/common_runtime/gpu/gpu_device.cc:985] 0:   N 
2019-03-09 01:58:32.567335: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1098] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3045 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
2019-03-09 01:58:32.567671: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1485] Adding visible gpu devices: 0
2019-03-09 01:58:32.567690: I tensorflow/core/common_runtime/gpu/gpu_device.cc:966] Device interconnect StreamExecutor with strength 1 edge matrix:
2019-03-09 01:58:32.567709: I tensorflow/core/common_runtime/gpu/gpu_device.cc:972]      0 
2019-03-09 01:58:32.567713: I tensorflow/core/common_runtime/gpu/gpu_device.cc:985] 0:   N 
2019-03-09 01:58:32.567817: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1098] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 3045 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1050, pci bus id: 0000:01:00.0, compute capability: 6.1)
  created train model with fresh parameters, time 0.11s
  created infer model with fresh parameters, time 0.04s
  # 1008
    src: is garibaldi monument in taganrog in taganrog
    ref: ask where brack_open dbr_Garibaldi_Monument_in_Taganrog dbo_location dbr_Taganrog brack_close
    nmt: var_b dbr_Nehanda_Abiodun) dbr_Nehanda_Abiodun) dbr_Stacey_Michelsen dbr_Stacey_Michelsen dbr_Stacey_Michelsen dbr_Beijing dbr_Beijing dbr_Nehanda_Abiodun) dbr_Nehanda_Abiodun) dbr_Nehanda_Abiodun) dbr_University_of_Dundee dbr_University_of_Dundee dbr_Statue_of_Equality
  created eval model with fresh parameters, time 0.05s
  eval dev: perplexity 2529.12, time 0s, Sat Mar  9 01:58:33 2019.
  eval test: perplexity 2529.05, time 1s, Sat Mar  9 01:58:35 2019.
2019-03-09 01:58:35.238722: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:35.238722: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:35.238722: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
# Start step 0, lr 1, Sat Mar  9 01:58:35 2019
# Init train iterator, skipping 0 elements
# Finished an epoch, step 86. Perform external evaluation
2019-03-09 01:58:38.807351: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:38.807356: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:38.807406: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 753
    src: what are the coordinates of raymond's tomb
    ref: select var_a where brack_open dbr_Raymond's_Tomb georss_point var_a brack_close
    nmt: Colombian_statue dbr_Alaska:_The_Last_Frontier) dbr_Alaska:_The_Last_Frontier) dbr_St._Paul's_Church,_Manora dbr_St._Paul's_Church,_Manora dbr_St._Paul's_Church,_Manora dbr_St._Paul's_Church,_Manora dbr_St._Paul's_Church,_Manora dbr_University_of_Portsmouth dbr_Urasoe,_Okinawa dbr_Urasoe,_Okinawa dbr_Urasoe,_Okinawa dbr_Urasoe,_Okinawa dbr_Urasoe,_Okinawa
2019-03-09 01:58:38.863814: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:38.863814: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:38.863814: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  step 100 lr 1 step-time 0.04s wps 55.84K ppl 40.29 gN 8.08 bleu 0.00, Sat Mar  9 01:58:39 2019
# Finished an epoch, step 172. Perform external evaluation
2019-03-09 01:58:41.711908: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:41.711922: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:41.711940: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 905
    src: longitude of admiral david g. farragut
    ref: select var_a where brack_open dbr_Admiral_David_G._Farragut_(Ream_statue) geo_long var_a brack_close
    nmt: dbr_Richard_Hancox dbr_The_X_Factor_ Pyongyang dbr_Puspalal_Sharma) Pyongyang Pyongyang dbr_Puspalal_Sharma) dbr_St_George's_Park,_Port_Elizabeth dbr_St_George's_Park,_Port_Elizabeth dbr_St_George's_Park,_Port_Elizabeth dbr_St_George's_Park,_Port_Elizabeth math_gt
2019-03-09 01:58:41.765045: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:41.765048: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:41.765048: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  step 200 lr 1 step-time 0.03s wps 75.78K ppl 7.33 gN 4.94 bleu 0.00, Sat Mar  9 01:58:42 2019
# Finished an epoch, step 258. Perform external evaluation
2019-03-09 01:58:44.691675: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:44.691675: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:44.691712: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 782
    src: what are the coordinates of arch of triumph
    ref: select var_a where brack_open dbr_Arch_of_Triumph_(Pyongyang) georss_point var_a brack_close
    nmt: dbr_Admiral_David_G._Farragut_(Ream_statue) dbr_Stuart_Collection) dbo_Plant dbr_Al-Qoubaiyat dbr_Sandakan_Massacre_Memorial dbr_Ryūzu_Falls dbr_Ryūzu_Falls dbr_Ryūzu_Falls dbr_Tivadar_Puskás dbr_Tivadar_Puskás dbr_1820_Settlers_National_Monument dbr_1820_Settlers_National_Monument footballer footballer footballer footballer
2019-03-09 01:58:44.745800: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:44.745803: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:44.745803: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  step 300 lr 1 step-time 0.03s wps 73.88K ppl 4.28 gN 3.89 bleu 0.00, Sat Mar  9 01:58:46 2019
# Finished an epoch, step 344. Perform external evaluation
2019-03-09 01:58:47.574674: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:47.574674: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:47.574683: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 155
    src: major general george b. mcclellan
    ref: select var_a where brack_open dbr_Major_General_George_B._McClellan dbo_abstract var_a brack_close
    nmt: dbr_Stuart_Island_(Washington)) dbr_United_States_congressional_delegations_from_Alabama) dbr_United_States_congressional_delegations_from_Alabama) dbr_Heiner_Moraing) dbr_Heiner_Moraing) dbr_Heiner_Moraing) dbr_Heiner_Moraing) dbr_Japan_women's_national_handball_team) dbr_Japan_women's_national_handball_team) dbr_Japan_women's_national_handball_team)
2019-03-09 01:58:47.626518: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:47.626518: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:47.626525: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  step 400 lr 1 step-time 0.03s wps 76.82K ppl 3.48 gN 2.88 bleu 0.00, Sat Mar  9 01:58:49 2019
# Finished an epoch, step 430. Perform external evaluation
2019-03-09 01:58:50.428250: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:50.428254: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:50.428290: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 849
    src: how north is nagasaki peace park
    ref: select var_a where brack_open dbr_Nagasaki_Peace_Park geo_lat var_a brack_close
    nmt: dbr_Monte dbr_Austin_Peck) dbr_Austin_Peck) dbr_1966_New_York_City_smog) dbr_1966_New_York_City_smog) dbr_Manod_Mawr dbr_Manod_Mawr dbr_Sivrihisar dbr_Sivrihisar dbr_1990–91_Divizia_B) dbr_Queen's_University_Belfast dbr_Billionth_Barrel_Monument
2019-03-09 01:58:50.481860: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:50.481871: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:50.481860: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  created infer model with fresh parameters, time 0.04s
  step 500 lr 1 step-time 0.03s wps 75.73K ppl 3.06 gN 2.98 bleu 0.00, Sat Mar  9 01:58:52 2019
# Finished an epoch, step 516. Perform external evaluation
2019-03-09 01:58:53.350931: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:53.350952: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:53.350978: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 422
    src: give me the location of alabama state monument
    ref: select var_a where brack_open dbr_Alabama_State_Monument dbo_location var_a brack_close
    nmt: dbr_Indonesia) dbr_Indonesia) dbr_Pioneer_Monument_(San_Francisco) dbr_Peter_Dedevbo) dbr_Peter_Dedevbo) dbr_Jay_Hammond dbr_Arturo_Gatti_vs._Floyd_Mayweather_Jr. dbr_Arturo_Gatti_vs._Floyd_Mayweather_Jr. dbr_Arturo_Gatti_vs._Floyd_Mayweather_Jr. dbr_Peter_Dedevbo) dbr_Peter_Dedevbo) dbr_Arturo_Gatti_vs._Floyd_Mayweather_Jr. dbr_Arturo_Gatti_vs._Floyd_Mayweather_Jr. dbr_Tomb_of_Caecilia_Metella dbr_Tomb_of_Caecilia_Metella dbr_Tomb_of_Caecilia_Metella
2019-03-09 01:58:53.416735: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:53.416735: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:53.416735: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  step 600 lr 1 step-time 0.03s wps 68.84K ppl 2.90 gN 2.48 bleu 0.00, Sat Mar  9 01:58:56 2019
# Finished an epoch, step 602. Perform external evaluation
2019-03-09 01:58:56.522482: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:56.522487: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:56.522492: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 570
    src: what is warsaw uprising monument related to
    ref: select var_a where brack_open dbr_Warsaw_Uprising_Monument dct_subject var_a brack_close
    nmt: dbr_Tim_Lips) dbr_Tim_Lips) dbr_Gwalior_Monument) dbr_Gwalior_Monument) dbr_Plymouth_Hoe dbr_Plymouth_Hoe dbr_Madhya_Pradesh dbr_Madhya_Pradesh dbr_Derrick_Atkins) dbr_Derrick_Atkins) dbr_Joseph_Smith_III) dbr_Joseph_Smith_III) dbr_Butt-Millet_Memorial_Fountain dbr_Butt-Millet_Memorial_Fountain
2019-03-09 01:58:56.584631: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:56.584651: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:56.584651: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
# Finished an epoch, step 688. Perform external evaluation
2019-03-09 01:58:59.679032: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:59.679032: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:59.679038: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  # 1242
    src: give me the willamette river tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Willamette_River
    nmt: dbr_Origin_of_the_Moon dbr_Château_de_l'Horizon wildcard wildcard wildcard dbr_Nakhchivan_(city) dbr_Nakhchivan_(city) dbr_Nakhchivan_(city) dbr_Barrière_d’Enfer dbr_Barrière_d’Enfer Colombian_statue dbr_Gibson_Les_Paul) dbr_Gibson_Les_Paul) dbr_Gibson_Les_Paul)
2019-03-09 01:58:59.740919: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:58:59.740924: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:58:59.740924: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  step 700 lr 1 step-time 0.04s wps 60.67K ppl 2.73 gN 2.74 bleu 0.00, Sat Mar  9 01:59:00 2019
# Finished an epoch, step 774. Perform external evaluation
2019-03-09 01:59:02.731159: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:02.731186: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:02.731189: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  # 109
    src: what is fortifications on the caribbean side of panama: portobelo-san lorenzo
    ref: select var_a where brack_open dbr_Fortifications_on_the_Caribbean_Side_of_Panama:_Portobelo-San_Lorenzo dbo_abstract var_a brack_close
    nmt: dbr_Liberty_Monument_ dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame dbr_Alpes-Maritimes dbr_Alpes-Maritimes dbr_Alpes-Maritimes dbr_Khojaly_massacre_memorials dbr_Lahore Guangzhou Guangzhou dbr_Tomb_of_National_Heroes_(Ljubljana) dbr_Tomb_of_National_Heroes_(Ljubljana) dbr_PPG_Paints_Arena dbr_PPG_Paints_Arena dbr_PPG_Paints_Arena dbr_University_of_Central_Lancashire dbr_University_of_Central_Lancashire dbr_University_of_Central_Lancashire dbr_Pioneers'_Obelisk_ dbr_Pioneers'_Obelisk_ dbr_Pioneers'_Obelisk_ dbr_Vijayawada,_India dbr_Vijayawada,_India
2019-03-09 01:59:02.797610: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:02.797610: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:02.797612: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  step 800 lr 1 step-time 0.03s wps 72.82K ppl 2.51 gN 1.89 bleu 0.00, Sat Mar  9 01:59:03 2019
# Finished an epoch, step 860. Perform external evaluation
2019-03-09 01:59:05.746047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:05.746061: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:05.746063: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.04s
  # 146
    src: kundasang war memorial
    ref: select var_a where brack_open dbr_Kundasang_War_Memorial dbo_abstract var_a brack_close
    nmt: dbr_New_Netherland dbr_Warsaw_Ghetto_boundary_markers dbr_Warsaw_Ghetto_boundary_markers dbr_Warsaw_Ghetto_boundary_markers dbr_Monument_to_the_Battle_of_Monte_Cassino_in_Warsaw dbr_Monument_to_the_Battle_of_Monte_Cassino_in_Warsaw
2019-03-09 01:59:05.801407: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:05.801407: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:05.801409: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  created infer model with fresh parameters, time 0.05s
  step 900 lr 1 step-time 0.03s wps 72.75K ppl 2.37 gN 1.46 bleu 0.00, Sat Mar  9 01:59:07 2019
# Finished an epoch, step 946. Perform external evaluation
2019-03-09 01:59:08.749555: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:08.749571: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:08.749579: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  created infer model with fresh parameters, time 0.05s
  # 1101
    src: which is taller between newkirk viaduct monument and catacombs of kom el shoqafa
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Catacombs_of_Kom_el_Shoqafa) brack_close _oba_ var_b limit 1
    nmt: dbr_Queen_Margaret_University dbr_Queen_Margaret_University dbr_Queen_Margaret_University dbr_Musolaphone) dbr_Monument_to_Nizami_Ganjavi_in_Baku dbr_Monument_to_Nizami_Ganjavi_in_Baku dbr_Civil_War_Memorial_(Webster,_Massachusetts) dbr_Civil_War_Memorial_(Webster,_Massachusetts) dbr_Drug_development) dbr_University_of_Huddersfield dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Capsicum_plaster dbr_Cayucupil dbr_Cayucupil dbr_Cayucupil dbr_Cayucupil dbr_Cayucupil dbr_Cayucupil dbr_Memorial_for_the_victims_killed_by_OUN-UPA_ dbr_Memorial_for_the_victims_killed_by_OUN-UPA_ dbr_Memorial_for_the_victims_killed_by_OUN-UPA_
2019-03-09 01:59:08.819210: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:08.819223: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:08.819213: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  created infer model with fresh parameters, time 0.05s
  step 1000 lr 1 step-time 0.03s wps 73.29K ppl 2.39 gN 1.76 bleu 0.00, Sat Mar  9 01:59:10 2019
# Save eval, global step 1000
2019-03-09 01:59:10.937352: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:10.937352: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:10.937352: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 418
    src: give me the location of shirvanshahs’ bath houses
    ref: select var_a where brack_open dbr_Shirvanshahs’_bath_houses dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Aaron_Smith_(magician)) brack_close
2019-03-09 01:59:10.978928: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:10.978928: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:10.978928: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.03s
  eval dev: perplexity 2.33, time 0s, Sat Mar  9 01:59:11 2019.
  eval test: perplexity 2.36, time 1s, Sat Mar  9 01:59:12 2019.
# Finished an epoch, step 1032. Perform external evaluation
2019-03-09 01:59:13.761308: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:13.761308: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:13.761311: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 11
    src: is monument to cuauhtémoc a monument
    ref: ask where brack_open dbr_Monument_to_Cuauhtémoc rdf_type dbo_Monument brack_close
    nmt: ask where brack_open rdf_type rdf_type dbo_Monument brack_close
2019-03-09 01:59:13.790420: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:13.790420: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:13.790420: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 01:59:14 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 01:59:21 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1100 lr 1 step-time 0.03s wps 72.33K ppl 2.22 gN 1.34 bleu 32.74, Sat Mar  9 01:59:24 2019
# Finished an epoch, step 1118. Perform external evaluation
2019-03-09 01:59:25.175845: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:25.175868: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:25.175868: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1203
    src: which is longer houston police officer's memorial or 1948 arab–israeli war
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Houston_Police_Officer's_Memorial || var_a = dbr_1948_Arab–Israeli_War) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Monumento_a_los_heroes_de_El_Polvorín_(tomb) || var_a = dbr_Aaron_Smith_(magician)) brack_close _oba_ var_b limit 1
2019-03-09 01:59:25.212809: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:25.212818: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:25.212818: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 01:59:26 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 01:59:32 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1200 lr 1 step-time 0.03s wps 72.44K ppl 2.21 gN 1.09 bleu 32.74, Sat Mar  9 01:59:36 2019
# Finished an epoch, step 1204. Perform external evaluation
2019-03-09 01:59:36.544065: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:36.544065: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:36.544072: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 61
    src: is sensitive flame a monument
    ref: ask where brack_open dbr_Sensitive_flame rdf_type dbo_Monument brack_close
    nmt: ask where brack_open rdf_type rdf_type dbo_Monument brack_close
2019-03-09 01:59:36.573635: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:36.573637: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:36.573637: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 01:59:37 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 01:59:43 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 1290. Perform external evaluation
2019-03-09 01:59:47.924524: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:47.924535: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:47.924542: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 91
    src: what is quezon memorial shrine
    ref: select var_a where brack_open dbr_Quezon_Memorial_Shrine dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Garakopaktapa dbo_abstract var_a brack_close
2019-03-09 01:59:47.952977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:47.952977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:47.952977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 01:59:48 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 01:59:55 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1300 lr 1 step-time 0.04s wps 61.76K ppl 2.22 gN 1.45 bleu 32.74, Sat Mar  9 01:59:57 2019
# Finished an epoch, step 1376. Perform external evaluation
2019-03-09 01:59:59.290410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:59.290410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:59.290410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1163
    src: which is taller between ranevskaya monument and aboriginal peoples in canada
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Aboriginal_peoples_in_Canada) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Aaron_Smith_(magician)) brack_close _oba_ var_b limit 1
2019-03-09 01:59:59.327188: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 01:59:59.327188: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 01:59:59.327188: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:00 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:00:06 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1400 lr 1 step-time 0.03s wps 72.14K ppl 2.17 gN 1.19 bleu 32.74, Sat Mar  9 02:00:08 2019
# Finished an epoch, step 1462. Perform external evaluation
2019-03-09 02:00:10.736207: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:10.736207: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:10.736208: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 36
    src: is major general john a. logan a artwork
    ref: ask where brack_open dbr_Major_General_John_A._Logan rdf_type dbo_Artwork brack_close
    nmt: ask where brack_open dbr_Shot_at_Dawn_Memorial rdf_type dbo_Monument brack_close
2019-03-09 02:00:10.765901: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:10.765930: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:10.765930: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:11 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:00:18 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1500 lr 1 step-time 0.03s wps 70.80K ppl 2.08 gN 1.13 bleu 32.74, Sat Mar  9 02:00:20 2019
# Finished an epoch, step 1548. Perform external evaluation
2019-03-09 02:00:22.302330: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:22.302330: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:22.302332: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 279
    src: where can one find garibaldi monument in taganrog
    ref: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit attr_open
2019-03-09 02:00:22.338978: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:22.338982: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:22.338982: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:23 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:00:29 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1600 lr 1 step-time 0.03s wps 69.99K ppl 2.05 gN 1.03 bleu 32.74, Sat Mar  9 02:00:32 2019
# Finished an epoch, step 1634. Perform external evaluation
2019-03-09 02:00:34.042758: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:34.042770: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:34.042774: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1040
    src: is fortifications on the caribbean side of panama: portobelo-san lorenzo in portobelo
    ref: ask where brack_open dbr_Fortifications_on_the_Caribbean_Side_of_Panama:_Portobelo-San_Lorenzo dbo_location dbr_Portobelo brack_close
    nmt: ask where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_February_One || var_a = dbr_Aaron_Smith_(magician)) brack_close _oba_ var_b limit 1
2019-03-09 02:00:34.081547: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:34.081547: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:34.081555: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:34 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Sat Mar  9 02:00:42 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1700 lr 1 step-time 0.03s wps 71.85K ppl 2.03 gN 1.14 bleu 32.74, Sat Mar  9 02:00:45 2019
# Finished an epoch, step 1720. Perform external evaluation
2019-03-09 02:00:46.300289: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:46.300289: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:46.300299: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 247
    src: where is ramavarma appan thampuran memorial
    ref: select var_a where brack_open dbr_Ramavarma_Appan_Thampuran_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
2019-03-09 02:00:46.332764: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:46.332765: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:46.332765: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:47 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:00:53 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1800 lr 1 step-time 0.03s wps 71.28K ppl 2.00 gN 1.37 bleu 32.74, Sat Mar  9 02:00:57 2019
# Finished an epoch, step 1806. Perform external evaluation
2019-03-09 02:00:58.172617: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:58.172631: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:58.172649: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 511
    src: when was nicolaus copernicus monument completed
    ref: select var_a where brack_open dbr_Nicolaus_Copernicus_Monument,_Warsaw dbp_complete var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
2019-03-09 02:00:58.205394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:00:58.205394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:00:58.205394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:00:59 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:01:05 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 1892. Perform external evaluation
2019-03-09 02:01:10.087711: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:10.087711: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:10.087711: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 645
    src: what is baku turkish martyrs' memorial all about
    ref: select var_a where brack_open dbr_Baku_Turkish_Martyrs'_Memorial dct_subject var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit attr_open
2019-03-09 02:01:10.131554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:10.131554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:10.131554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.03s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:01:11 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:01:17 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 1900 lr 1 step-time 0.04s wps 60.01K ppl 2.00 gN 1.24 bleu 32.74, Sat Mar  9 02:01:19 2019
# Finished an epoch, step 1978. Perform external evaluation
2019-03-09 02:01:21.968873: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:21.968887: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:21.968882: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 260
    src: where is prussian national monument for the liberation wars
    ref: select var_a where brack_open dbr_Prussian_National_Monument_for_the_Liberation_Wars dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit attr_open
2019-03-09 02:01:22.002561: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:22.002561: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:22.002563: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:01:22 2019.
  bleu dev: 32.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:01:29 2019.
  bleu test: 32.2
  saving hparams to /tmp/nmt_model/hparams
  step 2000 lr 1 step-time 0.03s wps 73.32K ppl 1.90 gN 1.22 bleu 32.74, Sat Mar  9 02:01:31 2019
# Save eval, global step 2000
2019-03-09 02:01:31.967351: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:31.967355: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:31.967361: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1010
    src: is guanyin of the south china sea in china
    ref: ask where brack_open dbr_Guanyin_of_the_South_China_Sea,_Mount_Xiqiao dbo_location dbr_China brack_close
    nmt: ask where brack_open dbr_Paritala_Anjaneya_Temple dbo_location dbr_Azerbaijan brack_close
2019-03-09 02:01:31.999693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:31.999693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:31.999693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  eval dev: perplexity 2.26, time 0s, Sat Mar  9 02:01:32 2019.
  eval test: perplexity 2.15, time 1s, Sat Mar  9 02:01:33 2019.
# Finished an epoch, step 2064. Perform external evaluation
2019-03-09 02:01:35.534689: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:35.534702: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:35.534703: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 56
    src: is old father time a monument
    ref: ask where brack_open dbr_Old_Father_Time rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Khojaly_Massacre rdf_type dbo_Monument brack_close
2019-03-09 02:01:35.564504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:35.564504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:35.564504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:01:36 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:01:41 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2100 lr 1 step-time 0.03s wps 74.07K ppl 1.88 gN 1.52 bleu 57.51, Sat Mar  9 02:01:43 2019
# Finished an epoch, step 2150. Perform external evaluation
2019-03-09 02:01:45.337968: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:45.337987: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:45.337987: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 324
    src: location of grotto of our lady of lourdes
    ref: select var_a where brack_open dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_The_Shot_in_the_Back dbo_location var_a brack_close
2019-03-09 02:01:45.368953: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:45.368953: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:45.368953: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:01:45 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:01:51 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2200 lr 1 step-time 0.03s wps 74.49K ppl 1.85 gN 1.22 bleu 57.51, Sat Mar  9 02:01:53 2019
# Finished an epoch, step 2236. Perform external evaluation
2019-03-09 02:01:54.958304: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:54.958304: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:54.958305: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 261
    src: where is mount royal cross
    ref: select var_a where brack_open dbr_Mount_Royal_Cross dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Garghabazar_Mosque dbo_location var_a brack_close
2019-03-09 02:01:54.988839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:01:54.988839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:01:54.988839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:01:55 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:02:00 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2300 lr 1 step-time 0.03s wps 72.59K ppl 1.76 gN 1.11 bleu 57.51, Sat Mar  9 02:02:04 2019
# Finished an epoch, step 2322. Perform external evaluation
2019-03-09 02:02:04.740086: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:04.740086: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:04.740090: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1250
    src: give me the diogo carvalho tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Diogo_Carvalho
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
2019-03-09 02:02:04.773565: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:04.773567: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:04.773568: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:02:05 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:02:10 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2400 lr 1 step-time 0.03s wps 74.71K ppl 1.75 gN 1.25 bleu 57.51, Sat Mar  9 02:02:14 2019
# Finished an epoch, step 2408. Perform external evaluation
2019-03-09 02:02:14.406376: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:14.406376: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:14.406381: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1181
    src: which is longer ahu akivi or jon foo
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Ahu_Akivi || var_a = dbr_Jon_Foo) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Wat's_Dyke || var_a = dbr_La_traviata) brack_close
2019-03-09 02:02:14.440895: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:14.440895: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:14.440895: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:02:15 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:02:20 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 2494. Perform external evaluation
2019-03-09 02:02:24.343470: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:24.343470: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:24.343470: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 158
    src: monument of liberty
    ref: select var_a where brack_open dbr_Monument_of_Liberty,_Chișinău dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Southern_General_Cemetery dbo_abstract
2019-03-09 02:02:24.373073: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:24.373073: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:24.373073: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:02:25 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:02:30 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2500 lr 1 step-time 0.04s wps 59.02K ppl 1.75 gN 1.43 bleu 57.51, Sat Mar  9 02:02:32 2019
# Finished an epoch, step 2580. Perform external evaluation
2019-03-09 02:02:34.844369: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:34.844379: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:34.844383: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 3
    src: is zar cave a monument
    ref: ask where brack_open dbr_Zar_Cave rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Imamzadeh_(Ganja) rdf_type dbo_Monument brack_close
2019-03-09 02:02:34.873800: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:34.873812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:34.873812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:02:35 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:02:41 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2600 lr 1 step-time 0.04s wps 67.10K ppl 1.67 gN 1.43 bleu 57.51, Sat Mar  9 02:02:43 2019
# Finished an epoch, step 2666. Perform external evaluation
2019-03-09 02:02:45.385280: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:45.385280: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:45.385298: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 976
    src: when was quezon memorial shrine built
    ref: select var_a where brack_open dbr_Quezon_Memorial_Shrine dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_People's_Friendship_Arch dbp_complete var_a brack_close
2019-03-09 02:02:45.416177: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:02:45.416177: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:02:45.416195: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:02:46 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:02:51 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2700 lr 1 step-time 0.07s wps 33.01K ppl 1.64 gN 1.14 bleu 57.51, Sat Mar  9 02:02:58 2019
# Finished an epoch, step 2752. Perform external evaluation
2019-03-09 02:03:02.579462: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:02.579462: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:02.579463: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.04s
  # 305
    src: where can one find southern general cemetery
    ref: select var_a where brack_open dbr_Southern_General_Cemetery dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Garghabazar_Mosque dbo_location var_a brack_close
2019-03-09 02:03:02.636112: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:02.636112: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:03:02.636116: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.04s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 1s, Sat Mar  9 02:03:03 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 8s, Sat Mar  9 02:03:13 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2800 lr 1 step-time 0.07s wps 34.60K ppl 1.58 gN 1.22 bleu 57.51, Sat Mar  9 02:03:16 2019
# Finished an epoch, step 2838. Perform external evaluation
2019-03-09 02:03:18.453709: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:03:18.453735: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:18.453716: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.03s
  # 941
    src: building date of hyde park holocaust memorial
    ref: select var_a where brack_open dbr_Hyde_Park_Holocaust_memorial dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Double_Six_Monument dbp_complete var_a brack_close
2019-03-09 02:03:18.497047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:18.497047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:18.497047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.03s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:03:19 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Sat Mar  9 02:03:26 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 2900 lr 1 step-time 0.04s wps 57.07K ppl 1.54 gN 1.26 bleu 57.51, Sat Mar  9 02:03:30 2019
# Finished an epoch, step 2924. Perform external evaluation
2019-03-09 02:03:31.369025: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:31.369025: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:03:31.369025: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 579
    src: what is ahmadalilar mausoleum related to
    ref: select var_a where brack_open dbr_Ahmadalilar_Mausoleum dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Ahu_Akivi dct_subject var_a brack_close
2019-03-09 02:03:31.406177: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:03:31.406179: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:31.406180: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:03:32 2019.
  bleu dev: 57.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:03:39 2019.
  bleu test: 56.9
  saving hparams to /tmp/nmt_model/hparams
  step 3000 lr 1 step-time 0.04s wps 66.11K ppl 1.52 gN 1.54 bleu 57.51, Sat Mar  9 02:03:43 2019
# Save eval, global step 3000
2019-03-09 02:03:43.203367: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:43.203367: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:43.203387: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 27
    src: is union soldiers and sailors monument a band
    ref: ask where brack_open dbr_Union_Soldiers_and_Sailors_Monument rdf_type dbo_Band brack_close
    nmt: ask where brack_open dbr_Monument_to_the_People's_Heroes rdf_type dbo_Monument brack_close
2019-03-09 02:03:43.242146: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:43.242146: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:43.242146: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.03s
  eval dev: perplexity 1.72, time 0s, Sat Mar  9 02:03:43 2019.
  eval test: perplexity 1.54, time 2s, Sat Mar  9 02:03:45 2019.
# Finished an epoch, step 3010. Perform external evaluation
2019-03-09 02:03:45.973199: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:45.973199: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:45.973203: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1216
    src: which is longer ahu akivi or barry
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Ahu_Akivi || var_a = dbr_Barry_(dog)) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Wat's_Dyke || var_a =
2019-03-09 02:03:46.006348: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:46.006348: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:46.006348: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:03:46 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:03:52 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 3096. Perform external evaluation
2019-03-09 02:03:56.616047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:56.616047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:56.616064: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 69
    src: is mausoleum of sheikh juneyd a place
    ref: ask where brack_open dbr_Mausoleum_of_Sheikh_Juneyd rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Tomb_of_Hayreddin_Barbarossa rdf_type dbo_Place brack_close
2019-03-09 02:03:56.651314: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:03:56.651318: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:03:56.651320: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:03:57 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:02 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3100 lr 1 step-time 0.04s wps 60.83K ppl 1.49 gN 1.46 bleu 63.89, Sat Mar  9 02:04:04 2019
# Finished an epoch, step 3182. Perform external evaluation
2019-03-09 02:04:06.767020: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:06.767021: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:06.767051: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 323
    src: location of the bull of navan
    ref: select var_a where brack_open dbr_The_Bull_of_Navan dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Tomb_of_Hayreddin_Barbarossa dbo_location var_a brack_close
2019-03-09 02:04:06.800454: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:06.800464: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:06.800464: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:07 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:12 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3200 lr 1 step-time 0.04s wps 66.80K ppl 1.42 gN 1.27 bleu 63.89, Sat Mar  9 02:04:14 2019
# Finished an epoch, step 3268. Perform external evaluation
2019-03-09 02:04:17.075941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:17.075977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:17.076015: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1322
    src: what do holocaust memorial and holocaust memorial have in common
    ref: select wildcard where brack_open brack_open dbr_Holocaust_Memorial,_Montevideo var_a var_b sep_dot dbr_Holocaust_Memorial,_Montevideo var_a var_b brack_close UNION brack_open brack_open dbr_Holocaust_Memorial,_Montevideo var_a var_b sep_dot dbr_Holocaust_Memorial,_Montevideo var_a var_b brack_close UNION brack_open var_c var_d dbr_Holocaust_Memorial,_Montevideo sep_dot var_c var_d dbr_Holocaust_Memorial,_Montevideo brack_close brack_close UNION brack_open var_c var_d dbr_Holocaust_Memorial,_Montevideo sep_dot var_c var_d dbr_Holocaust_Memorial,_Montevideo brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Namantar_Shahid_Smarak var_a var_b sep_dot dbr_World_Cup_Sculpture var_a var_b brack_close UNION brack_open brack_open dbr_Sverd_i_fjell var_a var_b sep_dot
2019-03-09 02:04:17.115606: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:17.115608: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:17.115612: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:17 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:23 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3300 lr 1 step-time 0.03s wps 69.41K ppl 1.41 gN 1.38 bleu 63.89, Sat Mar  9 02:04:25 2019
# Finished an epoch, step 3354. Perform external evaluation
2019-03-09 02:04:27.278042: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:27.278042: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:27.278059: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 646
    src: what is villa la reine jeanne all about
    ref: select var_a where brack_open dbr_Villa_La_Reine_Jeanne dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Mausoleum_of_Ruhollah_Khomeini dct_subject var_a brack_close
2019-03-09 02:04:27.312034: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:27.312034: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:27.312034: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:27 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:33 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3400 lr 1 step-time 0.03s wps 69.53K ppl 1.36 gN 1.39 bleu 63.89, Sat Mar  9 02:04:36 2019
# Finished an epoch, step 3440. Perform external evaluation
2019-03-09 02:04:37.600812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:37.600832: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:37.600832: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1045
    src: is royal naval division memorial in london
    ref: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_London brack_close
    nmt: ask where brack_open dbr_Yusif_ibn_Kuseyir_Mausoleum dbo_location dbr_Ufa brack_close
2019-03-09 02:04:37.634962: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:37.634962: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:37.634962: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:38 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:43 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3500 lr 1 step-time 0.03s wps 69.91K ppl 1.35 gN 1.39 bleu 63.89, Sat Mar  9 02:04:47 2019
# Finished an epoch, step 3526. Perform external evaluation
2019-03-09 02:04:48.163800: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:48.163800: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:48.163801: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1022
    src: is victims of communism memorial in washington
    ref: ask where brack_open dbr_Victims_of_Communism_Memorial dbo_location dbr_Washington,_D.C sep_dot brack_close
    nmt: ask where brack_open dbr_Tomb_of_Hayreddin_Barbarossa dbo_location dbr_Poland brack_close
2019-03-09 02:04:48.196131: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:48.196137: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:48.196137: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:48 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:04:54 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3600 lr 1 step-time 0.03s wps 69.21K ppl 1.33 gN 1.46 bleu 63.89, Sat Mar  9 02:04:57 2019
# Finished an epoch, step 3612. Perform external evaluation
2019-03-09 02:04:58.312364: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:58.312369: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:58.312385: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 398
    src: give me the location of alexander garden obelisk
    ref: select var_a where brack_open dbr_Alexander_Garden_Obelisk dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Sverd_i_fjell dbo_location var_a brack_close
2019-03-09 02:04:58.348047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:04:58.348047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:04:58.348050: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:04:59 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:05:04 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 3698. Perform external evaluation
2019-03-09 02:05:08.924768: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:08.924769: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:08.924789: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 751
    src: what are the coordinates of obelisk of axum
    ref: select var_a where brack_open dbr_Obelisk_of_Axum georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Namantar_Shahid_Smarak georss_point var_a brack_close
2019-03-09 02:05:08.960438: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:08.960441: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:08.960441: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:05:09 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:05:15 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3700 lr 1 step-time 0.04s wps 59.84K ppl 1.31 gN 1.47 bleu 63.89, Sat Mar  9 02:05:17 2019
# Finished an epoch, step 3784. Perform external evaluation
2019-03-09 02:05:19.715067: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:19.715067: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:19.715071: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 794
    src: latitude of livesey hall war memorial
    ref: select var_a where brack_open dbr_Livesey_Hall_War_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Tawau_Japanese_War_Memorial geo_lat var_a brack_close
2019-03-09 02:05:19.747502: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:19.747504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:19.747504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:05:20 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:05:26 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3800 lr 1 step-time 0.04s wps 65.39K ppl 1.28 gN 1.35 bleu 63.89, Sat Mar  9 02:05:28 2019
# Finished an epoch, step 3870. Perform external evaluation
2019-03-09 02:05:30.522529: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:30.522542: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:30.522554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1021
    src: is ramagrama stupa in nawalparasi district
    ref: ask where brack_open dbr_Ramagrama_stupa dbo_location dbr_Nawalparasi_District brack_close
    nmt: ask where brack_open dbr_Cristo_Rei,_Madeira dbo_location dbr_Azerbaijan brack_close
2019-03-09 02:05:30.557029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:30.557029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:30.557029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:05:31 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:05:37 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 3900 lr 1 step-time 0.04s wps 65.37K ppl 1.25 gN 1.17 bleu 63.89, Sat Mar  9 02:05:40 2019
# Finished an epoch, step 3956. Perform external evaluation
2019-03-09 02:05:41.995504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:41.995525: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:41.995525: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1181
    src: which is longer ahu akivi or jon foo
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Ahu_Akivi || var_a = dbr_Jon_Foo) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Wat's_Dyke || var_a = dbr_J._C._Daniel) brack_close
2019-03-09 02:05:42.035023: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:42.035023: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:42.035033: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:05:42 2019.
  bleu dev: 63.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:05:48 2019.
  bleu test: 67.8
  saving hparams to /tmp/nmt_model/hparams
  step 4000 lr 1 step-time 0.04s wps 66.58K ppl 1.23 gN 1.36 bleu 63.89, Sat Mar  9 02:05:51 2019
# Save eval, global step 4000
2019-03-09 02:05:51.532306: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:51.532306: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:51.532315: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 1045
    src: is royal naval division memorial in london
    ref: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_London brack_close
    nmt: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_Washington,_D.C sep_dot brack_close
2019-03-09 02:05:51.568650: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:51.568650: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:51.568652: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  eval dev: perplexity 1.38, time 0s, Sat Mar  9 02:05:51 2019.
  eval test: perplexity 1.21, time 1s, Sat Mar  9 02:05:53 2019.
# Finished an epoch, step 4042. Perform external evaluation
2019-03-09 02:05:54.885771: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:54.885782: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:54.885812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 650
    src: what is kentucky medal of honor memorial all about
    ref: select var_a where brack_open dbr_Kentucky_Medal_of_Honor_Memorial dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Kentucky_Medal_of_Honor_Memorial dct_subject var_a brack_close
2019-03-09 02:05:54.921577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:05:54.921583: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:05:54.921583: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:05:55 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:06:01 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4100 lr 1 step-time 0.04s wps 68.30K ppl 1.22 gN 1.30 bleu 79.91, Sat Mar  9 02:06:04 2019
# Finished an epoch, step 4128. Perform external evaluation
2019-03-09 02:06:05.320667: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:05.320679: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:05.320729: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 447
    src: where is alley of classics located in
    ref: select var_a where brack_open dbr_Alley_of_Classics,_Bălți dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Alley_of_Classics,_Bălți dbo_location var_a brack_close
2019-03-09 02:06:05.354893: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:05.354893: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:05.354893: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:06:05 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:06:11 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4200 lr 1 step-time 0.04s wps 65.71K ppl 1.20 gN 1.22 bleu 79.91, Sat Mar  9 02:06:15 2019
# Finished an epoch, step 4214. Perform external evaluation
2019-03-09 02:06:15.900570: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:15.900581: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:15.900601: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 687
    src: what is roman dmowski monument about
    ref: select var_a where brack_open dbr_Roman_Dmowski_Monument,_Warsaw dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Roman_Dmowski_Monument,_Warsaw dct_subject var_a brack_close
2019-03-09 02:06:15.939292: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:15.939309: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:15.939309: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.03s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:06:16 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:06:22 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4300 lr 1 step-time 0.04s wps 64.52K ppl 1.20 gN 1.41 bleu 79.91, Sat Mar  9 02:06:26 2019
# Finished an epoch, step 4300. Perform external evaluation
2019-03-09 02:06:26.866339: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:26.866348: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:26.866354: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 1003
    src: is villa la reine jeanne in provence-alpes-côte d’azur
    ref: ask where brack_open dbr_Villa_La_Reine_Jeanne dbo_location dbr_Provence-Alpes-Côte_d’Azur brack_close
    nmt: ask where brack_open dbr_Villa_La_Reine_Jeanne dbo_location dbr_Washington,_D.C sep_dot brack_close
2019-03-09 02:06:26.903467: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:26.903467: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:26.903531: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:06:27 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:06:34 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 4386. Perform external evaluation
2019-03-09 02:06:38.807791: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:38.807807: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:38.807791: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 230
    src: where is clock tower
    ref: select var_a where brack_open dbr_Clock_Tower,_Faisalabad dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Clock_Tower,_Faisalabad dbo_location var_a brack_close
2019-03-09 02:06:38.845036: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:38.845052: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:38.845052: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:06:39 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:06:45 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4400 lr 1 step-time 0.04s wps 54.35K ppl 1.17 gN 1.05 bleu 79.91, Sat Mar  9 02:06:47 2019
# Finished an epoch, step 4472. Perform external evaluation
2019-03-09 02:06:50.052122: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:50.052137: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:50.052122: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 621
    src: what is cristo rey related to
    ref: select var_a where brack_open dbr_Cristo_Rey_(Colombian_statue) dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Miła_18 dct_subject var_a brack_close
2019-03-09 02:06:50.087154: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:06:50.087156: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:06:50.087155: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:06:50 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:06:55 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4500 lr 1 step-time 0.04s wps 68.08K ppl 1.16 gN 1.10 bleu 79.91, Sat Mar  9 02:06:58 2019
# Finished an epoch, step 4558. Perform external evaluation
2019-03-09 02:07:00.422662: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:00.422667: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:00.422684: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 728
    src: what is monument to salavat yulaev about
    ref: select var_a where brack_open dbr_Monument_to_Salavat_Yulaev dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Vojvoda_Vuk dct_subject var_a brack_close
2019-03-09 02:07:00.458170: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:00.458170: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:00.458170: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:01 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:07:06 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4600 lr 1 step-time 0.04s wps 67.78K ppl 1.15 gN 0.98 bleu 79.91, Sat Mar  9 02:07:09 2019
# Finished an epoch, step 4644. Perform external evaluation
2019-03-09 02:07:11.283836: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:11.283836: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:11.283836: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.03s
  # 309
    src: where can one find alley of classics
    ref: select var_a where brack_open dbr_Alley_of_Classics,_Chișinău dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Alley_of_Classics,_Bălți dbo_location var_a brack_close
2019-03-09 02:07:11.321139: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:11.321149: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:11.321149: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:12 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:07:17 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4700 lr 1 step-time 0.04s wps 65.03K ppl 1.15 gN 1.10 bleu 79.91, Sat Mar  9 02:07:21 2019
# Finished an epoch, step 4730. Perform external evaluation
2019-03-09 02:07:22.198558: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:22.198558: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:22.198577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 1142
    src: which is taller between newkirk viaduct monument and avanibhajana pallaveshwaram temple
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Avanibhajana_Pallaveshwaram_temple) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Chekhov_Monument_in_Taganrog) brack_close _oba_ var_b limit 1
2019-03-09 02:07:22.244807: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:22.244807: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:22.244808: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:22 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:07:28 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4800 lr 1 step-time 0.04s wps 65.87K ppl 1.13 gN 0.88 bleu 79.91, Sat Mar  9 02:07:32 2019
# Finished an epoch, step 4816. Perform external evaluation
2019-03-09 02:07:33.142085: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:33.142085: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:33.142088: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 933
    src: longitude of general john a. rawlins
    ref: select var_a where brack_open dbr_General_John_A._Rawlins geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_General_John_A._Rawlins geo_long var_a brack_close
2019-03-09 02:07:33.177458: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:33.177458: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:33.177459: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:33 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:07:39 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 4900 lr 1 step-time 0.04s wps 66.85K ppl 1.13 gN 0.89 bleu 79.91, Sat Mar  9 02:07:43 2019
# Finished an epoch, step 4902. Perform external evaluation
2019-03-09 02:07:44.065181: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:44.065181: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:44.065181: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 901
    src: longitude of cristo rei
    ref: select var_a where brack_open dbr_Cristo_Rei,_Madeira geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Cristo_Rei,_Madeira geo_long var_a brack_close
2019-03-09 02:07:44.100029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:44.100029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:44.100029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:44 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:07:50 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 4988. Perform external evaluation
2019-03-09 02:07:54.956784: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:07:54.956791: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:54.956794: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 446
    src: where is mémorial des martyrs de la déportation located in
    ref: select var_a where brack_open dbr_Mémorial_des_Martyrs_de_la_Déportation dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Mémorial_des_Martyrs_de_la_Déportation dbo_location var_a brack_close
2019-03-09 02:07:54.995400: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:54.995400: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:07:54.995408: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:07:55 2019.
  bleu dev: 79.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:01 2019.
  bleu test: 86.8
  saving hparams to /tmp/nmt_model/hparams
  step 5000 lr 1 step-time 0.04s wps 56.16K ppl 1.13 gN 1.03 bleu 79.91, Sat Mar  9 02:08:03 2019
# Save eval, global step 5000
2019-03-09 02:08:03.798437: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:03.798437: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:03.798456: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 278
    src: where can one find diri baba mausoleum
    ref: select var_a where brack_open dbr_Diri_Baba_Mausoleum dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Diri_Baba_Mausoleum dbo_location var_a brack_close
2019-03-09 02:08:03.834539: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:03.834539: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:03.834539: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  eval dev: perplexity 1.32, time 0s, Sat Mar  9 02:08:04 2019.
  eval test: perplexity 1.14, time 1s, Sat Mar  9 02:08:05 2019.
2019-03-09 02:08:06.036969: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:06.036969: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:06.036973: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 33
    src: is sandakan memorial park a place
    ref: ask where brack_open dbr_Sandakan_Memorial_Park rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Sandakan_Memorial_Park rdf_type dbo_Place brack_close
2019-03-09 02:08:06.071297: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:06.071297: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:06.071297: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:06 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:12 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 5074. Perform external evaluation
2019-03-09 02:08:16.186382: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:16.186390: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:16.186402: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 33
    src: is sandakan memorial park a place
    ref: ask where brack_open dbr_Sandakan_Memorial_Park rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Sandakan_Memorial_Park rdf_type dbo_Place brack_close
2019-03-09 02:08:16.219693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:16.219693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:16.219693: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:16 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:22 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5100 lr 1 step-time 0.04s wps 68.67K ppl 1.12 gN 1.00 bleu 82.42, Sat Mar  9 02:08:24 2019
# Finished an epoch, step 5160. Perform external evaluation
2019-03-09 02:08:26.647941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:26.647953: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:26.647941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 1310
    src: what do monument indië-nederland and monument indië-nederland have in common
    ref: select wildcard where brack_open brack_open dbr_Monument_Indië-Nederland var_a var_b sep_dot dbr_Monument_Indië-Nederland var_a var_b brack_close UNION brack_open brack_open dbr_Monument_Indië-Nederland var_a var_b sep_dot dbr_Monument_Indië-Nederland var_a var_b brack_close UNION brack_open var_c var_d dbr_Monument_Indië-Nederland sep_dot var_c var_d dbr_Monument_Indië-Nederland brack_close brack_close UNION brack_open var_c var_d dbr_Monument_Indië-Nederland sep_dot var_c var_d dbr_Monument_Indië-Nederland brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Monument_Indië-Nederland attr_open Decatur,_Indiana attr_close attr_open dbr_Republic_Monument attr_close attr_open Amsterdam attr_close var_a var_b brack_close UNION brack_open
2019-03-09 02:08:26.694570: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:26.694570: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:26.694570: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:27 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:33 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5200 lr 1 step-time 0.04s wps 66.26K ppl 1.12 gN 0.96 bleu 82.42, Sat Mar  9 02:08:35 2019
# Finished an epoch, step 5246. Perform external evaluation
2019-03-09 02:08:37.369379: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:37.369380: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:37.369384: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 1323
    src: what do wesselényi monument and wesselényi monument have in common
    ref: select wildcard where brack_open brack_open dbr_Wesselényi_Monument var_a var_b sep_dot dbr_Wesselényi_Monument var_a var_b brack_close UNION brack_open brack_open dbr_Wesselényi_Monument var_a var_b sep_dot dbr_Wesselényi_Monument var_a var_b brack_close UNION brack_open var_c var_d dbr_Wesselényi_Monument sep_dot var_c var_d dbr_Wesselényi_Monument brack_close brack_close UNION brack_open var_c var_d dbr_Wesselényi_Monument sep_dot var_c var_d dbr_Wesselényi_Monument brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_National_Monument_ attr_open Amsterdam attr_close var_a var_b sep_dot dbr_National_Monument_ attr_open Amsterdam attr_close var_a var_b brack_close UNION
2019-03-09 02:08:37.410116: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:37.410116: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:37.410120: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:38 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:43 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5300 lr 1 step-time 0.04s wps 68.74K ppl 1.11 gN 0.94 bleu 82.42, Sat Mar  9 02:08:47 2019
# Finished an epoch, step 5332. Perform external evaluation
2019-03-09 02:08:48.096209: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:48.096209: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:48.096209: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 725
    src: what is monument des braves about
    ref: select var_a where brack_open dbr_Monument_des_Braves,_Shawinigan dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_des_Braves,_Shawinigan dct_subject var_a brack_close
2019-03-09 02:08:48.131577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:48.131577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:48.131577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:48 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:08:54 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5400 lr 1 step-time 0.03s wps 67.01K ppl 1.11 gN 0.86 bleu 82.42, Sat Mar  9 02:08:58 2019
# Finished an epoch, step 5418. Perform external evaluation
2019-03-09 02:08:58.798805: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:58.798848: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:58.798812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 834
    src: latitude of jam gadang
    ref: select var_a where brack_open dbr_Jam_Gadang geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Jam_Gadang geo_lat var_a brack_close
2019-03-09 02:08:58.834664: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:08:58.834666: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:08:58.834666: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:08:59 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:09:05 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5500 lr 1 step-time 0.04s wps 67.37K ppl 1.10 gN 0.87 bleu 82.42, Sat Mar  9 02:09:09 2019
# Finished an epoch, step 5504. Perform external evaluation
2019-03-09 02:09:09.798633: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:09.798635: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:09.798638: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 394
    src: give me the location of hôtel de villeroy
    ref: select var_a where brack_open dbr_Hôtel_de_Villeroy dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Hôtel_de_Villeroy dbo_location var_a brack_close
2019-03-09 02:09:09.834199: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:09.834199: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:09.834200: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:09:10 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:09:15 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 5590. Perform external evaluation
2019-03-09 02:09:20.445611: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:20.445615: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:20.445621: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 600
    src: what is battle of otlukbeli martyrs' monument related to
    ref: select var_a where brack_open dbr_Battle_of_Otlukbeli_Martyrs'_Monument dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Battle_of_Otlukbeli_Martyrs'_Monument dct_subject var_a brack_close
2019-03-09 02:09:20.480828: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:20.480828: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:20.480828: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:09:21 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:09:27 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5600 lr 1 step-time 0.04s wps 57.21K ppl 1.09 gN 0.71 bleu 82.42, Sat Mar  9 02:09:29 2019
# Finished an epoch, step 5676. Perform external evaluation
2019-03-09 02:09:31.775713: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:31.775717: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:31.775735: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 902
    src: longitude of wilfrid laurier memorial
    ref: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial geo_long var_a brack_close
2019-03-09 02:09:31.812505: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:31.812523: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:31.812506: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:09:32 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:09:38 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5700 lr 1 step-time 0.04s wps 66.85K ppl 1.09 gN 0.66 bleu 82.42, Sat Mar  9 02:09:41 2019
# Finished an epoch, step 5762. Perform external evaluation
2019-03-09 02:09:43.489795: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:43.489819: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:43.489795: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.03s
  # 1218
    src: which is longer marco zero or ramone mckenzie
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Marco_Zero_(São_Paulo) || var_a = dbr_Ramone_McKenzie) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Marco_Zero_(São_Paulo) || var_a = dbr_Indonesian_language) brack_close
2019-03-09 02:09:43.537972: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:43.537972: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:43.537973: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.03s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:09:44 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:09:50 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5800 lr 1 step-time 0.04s wps 61.83K ppl 1.08 gN 0.61 bleu 82.42, Sat Mar  9 02:09:53 2019
# Finished an epoch, step 5848. Perform external evaluation
2019-03-09 02:09:55.014711: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:55.014724: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:55.014730: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.03s
  # 591
    src: what is monument in memory of the fallen polish pilots in world war ii related to
    ref: select var_a where brack_open dbr_Monument_in_Memory_of_the_Fallen_Polish_Pilots_in_World_War_II dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_in_Memory_of_the_Fallen_Polish_Pilots_in_World_War_II dct_subject var_a brack_close
2019-03-09 02:09:55.056570: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:09:55.056573: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:09:55.056573: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.03s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:09:55 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:10:01 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 5900 lr 1 step-time 0.04s wps 65.05K ppl 1.11 gN 1.39 bleu 82.42, Sat Mar  9 02:10:05 2019
# Finished an epoch, step 5934. Perform external evaluation
2019-03-09 02:10:06.305481: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:06.305481: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:06.305538: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.03s
  # 286
    src: where can one find lifeboat memorial
    ref: select var_a where brack_open dbr_Lifeboat_Memorial,_Lytham dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Lifeboat_Memorial,_Lytham dbo_location var_a brack_close
2019-03-09 02:10:06.343892: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:06.343893: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:06.343893: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:10:07 2019.
  bleu dev: 82.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:10:13 2019.
  bleu test: 86.5
  saving hparams to /tmp/nmt_model/hparams
  step 6000 lr 1 step-time 0.04s wps 67.69K ppl 1.10 gN 1.09 bleu 82.42, Sat Mar  9 02:10:16 2019
# Save eval, global step 6000
2019-03-09 02:10:17.023358: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:17.023361: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:17.023389: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
  # 1274
    src: give me the west london line tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_West_London_Line
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Khojaly_massacre_memorials
2019-03-09 02:10:17.069839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:17.069839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:17.069839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
  eval dev: perplexity 1.29, time 0s, Sat Mar  9 02:10:17 2019.
  eval test: perplexity 1.09, time 1s, Sat Mar  9 02:10:19 2019.
# Finished an epoch, step 6020. Perform external evaluation
2019-03-09 02:10:19.862665: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:19.862665: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:19.862667: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 330
    src: location of freedom monument
    ref: select var_a where brack_open dbr_Freedom_Monument,_Bydgoszcz dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Freedom_Monument,_Bydgoszcz dbo_location var_a brack_close
2019-03-09 02:10:19.899823: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:19.899828: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:19.899828: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:10:20 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:10:26 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6100 lr 1 step-time 0.04s wps 65.74K ppl 1.08 gN 0.74 bleu 84.58, Sat Mar  9 02:10:30 2019
# Finished an epoch, step 6106. Perform external evaluation
2019-03-09 02:10:31.130632: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:31.130648: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:31.130645: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
  # 1155
    src: which is taller between newkirk viaduct monument and j. c. daniel
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_J._C._Daniel) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Monument_to_Cuauhtémoc) brack_close _oba_ var_b limit 1
2019-03-09 02:10:31.179436: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:31.179436: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:31.179441: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:10:31 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:10:38 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 6192. Perform external evaluation
2019-03-09 02:10:42.508542: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:42.508542: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:42.508543: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 435
    src: where is monument to feodor chaliapin located in
    ref: select var_a where brack_open dbr_Monument_to_Feodor_Chaliapin dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Feodor_Chaliapin dbo_location var_a brack_close
2019-03-09 02:10:42.544603: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:42.544611: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:42.544611: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:10:43 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:10:49 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6200 lr 1 step-time 0.04s wps 56.66K ppl 1.08 gN 0.61 bleu 84.58, Sat Mar  9 02:10:50 2019
# Finished an epoch, step 6278. Perform external evaluation
2019-03-09 02:10:53.677840: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:10:53.677840: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:53.677840: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 852
    src: how north is victims of communism memorial
    ref: select var_a where brack_open dbr_Victims_of_Communism_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Victims_of_Communism_Memorial geo_lat var_a brack_close
2019-03-09 02:10:53.716526: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:53.716526: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:10:53.716526: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:10:54 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:11:00 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6300 lr 1 step-time 0.04s wps 62.66K ppl 1.07 gN 0.52 bleu 84.58, Sat Mar  9 02:11:02 2019
# Finished an epoch, step 6364. Perform external evaluation
2019-03-09 02:11:04.664993: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:04.664993: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:04.665020: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 871
    src: how north is player of backgammon
    ref: select var_a where brack_open dbr_Player_of_backgammon_(monument) geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Player_of_backgammon_(monument) geo_lat var_a brack_close
2019-03-09 02:11:04.699190: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:04.699190: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:04.699190: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:11:05 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:11:11 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6400 lr 1 step-time 0.04s wps 66.84K ppl 1.07 gN 0.56 bleu 84.58, Sat Mar  9 02:11:14 2019
# Finished an epoch, step 6450. Perform external evaluation
2019-03-09 02:11:16.118430: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:16.118430: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:16.118430: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
  # 60
    src: is husainabad clock tower a building
    ref: ask where brack_open dbr_Husainabad_Clock_Tower rdf_type dbo_Building brack_close
    nmt: ask where brack_open dbr_Husainabad_Clock_Tower rdf_type dbo_Building brack_close
2019-03-09 02:11:16.158811: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:16.158811: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:16.158813: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:11:16 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:11:22 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6500 lr 1 step-time 0.04s wps 65.13K ppl 1.07 gN 0.56 bleu 84.58, Sat Mar  9 02:11:26 2019
# Finished an epoch, step 6536. Perform external evaluation
2019-03-09 02:11:27.518087: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:27.518089: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:27.518137: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 312
    src: location of khanegah tomb
    ref: select var_a where brack_open dbr_Khanegah_tomb dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Khanegah_tomb dbo_location var_a brack_close
2019-03-09 02:11:27.556349: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:27.556349: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:27.556355: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:11:28 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:11:34 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6600 lr 1 step-time 0.04s wps 66.78K ppl 1.07 gN 0.55 bleu 84.58, Sat Mar  9 02:11:37 2019
# Finished an epoch, step 6622. Perform external evaluation
2019-03-09 02:11:38.539015: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:38.539028: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:38.539034: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
  # 22
    src: is monument to sergei yesenin in st. petersburg a monument
    ref: ask where brack_open dbr_Monument_to_Sergei_Yesenin_in_St._Petersburg rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Monument_to_Sergei_Yesenin_in_St._Petersburg rdf_type dbo_Monument brack_close
2019-03-09 02:11:38.576029: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:38.576031: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:38.576032: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:11:39 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:11:45 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6700 lr 1 step-time 0.04s wps 68.04K ppl 1.07 gN 0.65 bleu 84.58, Sat Mar  9 02:11:49 2019
# Finished an epoch, step 6708. Perform external evaluation
2019-03-09 02:11:49.522542: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:49.522546: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:11:49.522580: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 249
    src: where is kirna mausoleum
    ref: select var_a where brack_open dbr_Kirna_mausoleum dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Kirna_mausoleum dbo_location var_a brack_close
2019-03-09 02:11:49.555447: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:49.555447: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:11:49.555447: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:11:50 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:11:55 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 6794. Perform external evaluation
2019-03-09 02:12:00.519905: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:00.519918: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:00.519905: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 862
    src: how north is farhad and shirin monument
    ref: select var_a where brack_open dbr_Farhad_and_Shirin_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Farhad_and_Shirin_Monument geo_lat var_a brack_close
2019-03-09 02:12:00.554964: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:00.554964: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:00.554964: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:12:01 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:12:07 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6800 lr 1 step-time 0.04s wps 56.15K ppl 1.07 gN 0.60 bleu 84.58, Sat Mar  9 02:12:09 2019
# Finished an epoch, step 6880. Perform external evaluation
2019-03-09 02:12:12.242692: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:12.242696: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:12.242729: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 17
    src: is louis cyr monument a monument
    ref: ask where brack_open dbr_Louis_Cyr_Monument rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Louis_Cyr_Monument rdf_type dbo_Monument brack_close
2019-03-09 02:12:12.276441: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:12.276441: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:12.276453: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:12:13 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:12:18 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 6900 lr 1 step-time 0.04s wps 64.47K ppl 1.06 gN 0.50 bleu 84.58, Sat Mar  9 02:12:21 2019
# Finished an epoch, step 6966. Perform external evaluation
2019-03-09 02:12:23.387778: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:23.387779: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:23.387780: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 548
    src: what's treaty of lausanne monument and museum native name
    ref: select var_a where brack_open dbr_Treaty_of_Lausanne_Monument_and_Museum dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Treaty_of_Lausanne_Monument_and_Museum dbp_nativeName var_a brack_close
2019-03-09 02:12:23.423481: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:23.423481: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:23.423482: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:12:24 2019.
  bleu dev: 84.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:12:29 2019.
  bleu test: 93.6
  saving hparams to /tmp/nmt_model/hparams
  step 7000 lr 1 step-time 0.04s wps 66.64K ppl 1.06 gN 0.47 bleu 84.58, Sat Mar  9 02:12:32 2019
# Save eval, global step 7000
2019-03-09 02:12:32.953839: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:32.953867: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:32.953965: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 557
    src: what's dai kannon of kita no miyako park native name
    ref: select var_a where brack_open dbr_Dai_Kannon_of_Kita_no_Miyako_park dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Dai_Kannon_of_Kita_no_Miyako_park dbp_nativeName var_a brack_close
2019-03-09 02:12:32.993378: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:32.993378: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:32.993380: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:12:33 2019.
  eval test: perplexity 1.07, time 1s, Sat Mar  9 02:12:35 2019.
# Finished an epoch, step 7052. Perform external evaluation
2019-03-09 02:12:36.841110: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:36.841118: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:36.841133: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 332
    src: location of convento de los agustinos
    ref: select var_a where brack_open dbr_Convento_de_los_Agustinos dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Convento_de_los_Agustinos dbo_location var_a brack_close
2019-03-09 02:12:36.877452: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:36.877455: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:36.877455: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:12:37 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:12:43 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7100 lr 1 step-time 0.05s wps 52.08K ppl 1.06 gN 0.44 bleu 86.24, Sat Mar  9 02:12:47 2019
# Finished an epoch, step 7138. Perform external evaluation
2019-03-09 02:12:49.684749: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:49.684755: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:49.684755: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 308
    src: where can one find dupont circle fountain
    ref: select var_a where brack_open dbr_Dupont_Circle_Fountain dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Dupont_Circle_Fountain dbo_location var_a brack_close
2019-03-09 02:12:49.732483: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:12:49.732483: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:12:49.732522: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:12:50 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Sat Mar  9 02:12:57 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7200 lr 1 step-time 0.04s wps 56.67K ppl 1.06 gN 0.46 bleu 86.24, Sat Mar  9 02:13:01 2019
# Finished an epoch, step 7224. Perform external evaluation
2019-03-09 02:13:02.477423: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:02.477432: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:02.477445: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 983
    src: when was long live the victory of mao zedong thought built
    ref: select var_a where brack_open dbr_Long_Live_the_Victory_of_Mao_Zedong_Thought dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Long_Live_the_Victory_of_Mao_Zedong_Thought dbp_complete var_a brack_close
2019-03-09 02:13:02.523645: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:02.523660: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:02.523647: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:13:03 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:13:09 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7300 lr 1 step-time 0.04s wps 65.07K ppl 1.06 gN 0.58 bleu 86.24, Sat Mar  9 02:13:13 2019
# Finished an epoch, step 7310. Perform external evaluation
2019-03-09 02:13:14.032397: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:14.032397: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:14.032401: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 826
    src: latitude of billionth barrel monument
    ref: select var_a where brack_open dbr_Billionth_Barrel_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Billionth_Barrel_Monument geo_lat var_a brack_close
2019-03-09 02:13:14.074022: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:14.074024: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:14.074024: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:13:14 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:13:21 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 7396. Perform external evaluation
2019-03-09 02:13:25.488749: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:25.488757: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:25.488800: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 664
    src: what is major general george b. mcclellan all about
    ref: select var_a where brack_open dbr_Major_General_George_B._McClellan dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Major_General_George_B._McClellan dct_subject var_a brack_close
2019-03-09 02:13:25.526263: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:25.526263: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:25.526268: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:13:26 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:13:32 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7400 lr 1 step-time 0.04s wps 56.87K ppl 1.06 gN 0.53 bleu 86.24, Sat Mar  9 02:13:33 2019
# Finished an epoch, step 7482. Perform external evaluation
2019-03-09 02:13:36.739246: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:36.739246: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:36.739246: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 1045
    src: is royal naval division memorial in london
    ref: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_London brack_close
    nmt: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_Lancaster,_Lancashire brack_close
2019-03-09 02:13:36.777614: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:36.777634: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:36.777614: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:13:37 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:13:44 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7500 lr 1 step-time 0.04s wps 60.90K ppl 1.05 gN 0.44 bleu 86.24, Sat Mar  9 02:13:46 2019
# Finished an epoch, step 7568. Perform external evaluation
2019-03-09 02:13:48.591386: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:48.591387: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:48.591433: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 1279
    src: what do georgia guidestones and georgia guidestones have in common
    ref: select wildcard where brack_open brack_open dbr_Georgia_Guidestones var_a var_b sep_dot dbr_Georgia_Guidestones var_a var_b brack_close UNION brack_open brack_open dbr_Georgia_Guidestones var_a var_b sep_dot dbr_Georgia_Guidestones var_a var_b brack_close UNION brack_open var_c var_d dbr_Georgia_Guidestones sep_dot var_c var_d dbr_Georgia_Guidestones brack_close brack_close UNION brack_open var_c var_d dbr_Georgia_Guidestones sep_dot var_c var_d dbr_Georgia_Guidestones brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Buzeyir_cave var_a var_b sep_dot dbr_Buzeyir_cave var_a var_b brack_close UNION brack_open brack_open dbr_Buzeyir_cave var_a var_b sep_dot
2019-03-09 02:13:48.639491: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:13:48.639491: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:13:48.639491: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:13:49 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:13:56 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7600 lr 1 step-time 0.04s wps 61.64K ppl 1.05 gN 0.50 bleu 86.24, Sat Mar  9 02:13:59 2019
# Finished an epoch, step 7654. Perform external evaluation
2019-03-09 02:14:01.260538: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:01.260568: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:01.260569: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 415
    src: give me the location of carew cross
    ref: select var_a where brack_open dbr_Carew_Cross dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Carew_Cross dbo_location var_a brack_close
2019-03-09 02:14:01.301460: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:01.301464: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:01.301464: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:14:02 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:14:08 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7700 lr 1 step-time 0.04s wps 58.28K ppl 1.05 gN 0.47 bleu 86.24, Sat Mar  9 02:14:11 2019
# Finished an epoch, step 7740. Perform external evaluation
2019-03-09 02:14:12.825606: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:12.825627: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:12.825627: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 516
    src: when was wat's dyke completed
    ref: select var_a where brack_open dbr_Wat's_Dyke dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Offa's_Dyke dbp_complete var_a brack_close
2019-03-09 02:14:12.865885: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:12.865886: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:12.865892: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:14:13 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:14:19 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7800 lr 1 step-time 0.04s wps 63.95K ppl 1.05 gN 0.41 bleu 86.24, Sat Mar  9 02:14:23 2019
# Finished an epoch, step 7826. Perform external evaluation
2019-03-09 02:14:24.062480: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:24.062496: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:24.062488: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 794
    src: latitude of livesey hall war memorial
    ref: select var_a where brack_open dbr_Livesey_Hall_War_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Livesey_Hall_War_Memorial geo_lat var_a brack_close
2019-03-09 02:14:24.102462: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:24.102462: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:24.102463: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:14:24 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:14:31 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 7900 lr 1 step-time 0.04s wps 64.48K ppl 1.05 gN 0.42 bleu 86.24, Sat Mar  9 02:14:35 2019
# Finished an epoch, step 7912. Perform external evaluation
2019-03-09 02:14:35.871494: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:35.871496: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:35.871505: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 670
    src: what is juma mosque in sheki all about
    ref: select var_a where brack_open dbr_Juma_Mosque_in_Sheki dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Juma_Mosque_in_Sheki dct_subject var_a brack_close
2019-03-09 02:14:35.913808: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:35.913808: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:35.913810: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:14:36 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:14:42 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 7998. Perform external evaluation
2019-03-09 02:14:47.454063: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:47.454068: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:47.454074: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  # 271
    src: where can one find dewey monument
    ref: select var_a where brack_open dbr_Dewey_Monument dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Dewey_Monument dbo_location var_a brack_close
2019-03-09 02:14:47.498038: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:47.498063: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:47.498070: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:14:48 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:14:54 2019.
  bleu test: 94.6
  saving hparams to /tmp/nmt_model/hparams
  step 8000 lr 1 step-time 0.04s wps 55.72K ppl 1.05 gN 0.43 bleu 86.24, Sat Mar  9 02:14:55 2019
# Save eval, global step 8000
2019-03-09 02:14:56.202616: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:56.202779: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:56.202817: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.05s
  # 544
    src: what's damjili cave native name
    ref: select var_a where brack_open dbr_Damjili_Cave dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Damjili_Cave dbp_nativeName var_a brack_close
2019-03-09 02:14:56.275732: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:14:56.275850: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:14:56.275984: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.05s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:14:56 2019.
  eval test: perplexity 1.06, time 1s, Sat Mar  9 02:14:58 2019.
# Finished an epoch, step 8084. Perform external evaluation
2019-03-09 02:15:01.060765: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:01.060765: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:01.060794: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 299
    src: where can one find samuel hahnemann monument
    ref: select var_a where brack_open dbr_Samuel_Hahnemann_Monument dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Samuel_Hahnemann_Monument dbo_location var_a brack_close
2019-03-09 02:15:01.099862: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:01.099862: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:01.099868: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:01 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:15:07 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8100 lr 1 step-time 0.04s wps 63.83K ppl 1.05 gN 0.42 bleu 86.24, Sat Mar  9 02:15:09 2019
# Finished an epoch, step 8170. Perform external evaluation
2019-03-09 02:15:12.080786: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:12.080786: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:12.080810: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 188
    src: fort saint-elme
    ref: select var_a where brack_open dbr_Fort_Saint-Elme_(France) dbo_abstract var_a brack_close
    nmt: select var_a where brack_open
2019-03-09 02:15:12.113410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:12.113410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:12.113410: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:12 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Sat Mar  9 02:15:19 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8200 lr 1 step-time 0.04s wps 68.21K ppl 1.05 gN 0.47 bleu 86.24, Sat Mar  9 02:15:21 2019
# Finished an epoch, step 8256. Perform external evaluation
2019-03-09 02:15:23.548218: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:23.548270: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:23.548233: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 25
    src: is namantar shahid smarak a place
    ref: ask where brack_open dbr_Namantar_Shahid_Smarak rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Namantar_Shahid_Smarak rdf_type dbo_Place brack_close
2019-03-09 02:15:23.588729: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:23.588729: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:23.588729: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:24 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:15:30 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8300 lr 1 step-time 0.04s wps 66.12K ppl 1.05 gN 0.43 bleu 86.24, Sat Mar  9 02:15:33 2019
# Finished an epoch, step 8342. Perform external evaluation
2019-03-09 02:15:34.597355: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:34.597385: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:34.597386: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 1131
    src: which is taller between newkirk viaduct monument and kristian kostov
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Kristian_Kostov) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Judd_Nelson) brack_close _oba_ var_b limit 1
2019-03-09 02:15:34.643871: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:34.643871: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:34.643874: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:35 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:15:41 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8400 lr 1 step-time 0.04s wps 68.14K ppl 1.04 gN 0.42 bleu 86.24, Sat Mar  9 02:15:44 2019
# Finished an epoch, step 8428. Perform external evaluation
2019-03-09 02:15:45.529046: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:45.529049: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:45.529098: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 708
    src: what is chekhov monument in taganrog about
    ref: select var_a where brack_open dbr_Chekhov_Monument_in_Taganrog dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Chekhov_Monument_in_Taganrog dct_subject var_a brack_close
2019-03-09 02:15:45.564884: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:45.564884: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:45.564884: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:46 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:15:51 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8500 lr 1 step-time 0.03s wps 69.18K ppl 1.05 gN 0.45 bleu 86.24, Sat Mar  9 02:15:55 2019
# Finished an epoch, step 8514. Perform external evaluation
2019-03-09 02:15:56.128235: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:15:56.128259: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:56.128261: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 124
    src: what is mausoleum of ruhollah khomeini
    ref: select var_a where brack_open dbr_Mausoleum_of_Ruhollah_Khomeini dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Mausoleum_of_Ruhollah_Khomeini dbo_abstract var_a brack_close
2019-03-09 02:15:56.165394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:56.165394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:15:56.165394: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:15:56 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:02 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8600 lr 1 step-time 0.04s wps 66.46K ppl 1.04 gN 0.49 bleu 86.24, Sat Mar  9 02:16:07 2019
# Finished an epoch, step 8600. Perform external evaluation
2019-03-09 02:16:07.110605: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:07.110617: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:07.110630: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 933
    src: longitude of general john a. rawlins
    ref: select var_a where brack_open dbr_General_John_A._Rawlins geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_General_John_A._Rawlins geo_long var_a brack_close
2019-03-09 02:16:07.146110: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:07.146110: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:07.146110: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:16:07 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:13 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 8686. Perform external evaluation
2019-03-09 02:16:17.420641: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:17.420641: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:17.420642: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 94
    src: what is mount royal cross
    ref: select var_a where brack_open dbr_Mount_Royal_Cross dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Mount_Royal_Cross dbo_abstract var_a brack_close
2019-03-09 02:16:17.455485: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:17.455490: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:17.455490: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:16:18 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:23 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8700 lr 1 step-time 0.04s wps 58.68K ppl 1.04 gN 0.42 bleu 86.24, Sat Mar  9 02:16:25 2019
# Finished an epoch, step 8772. Perform external evaluation
2019-03-09 02:16:27.972575: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:27.972577: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:27.972598: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 1128
    src: which is taller between newkirk viaduct monument and 2012 new south wales swifts season
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_2012_New_South_Wales_Swifts_season) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Disappearance_of_Terrance_Williams_and_Felipe_Santos) brack_close _oba_ var_b limit 1
2019-03-09 02:16:28.017004: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:28.017008: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:28.017009: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:16:28 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:34 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8800 lr 1 step-time 0.04s wps 67.17K ppl 1.04 gN 0.45 bleu 86.24, Sat Mar  9 02:16:36 2019
# Finished an epoch, step 8858. Perform external evaluation
2019-03-09 02:16:38.510752: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:38.510772: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:38.510782: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 804
    src: latitude of world cup sculpture
    ref: select var_a where brack_open dbr_World_Cup_Sculpture geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_World_Cup_Sculpture geo_lat var_a brack_close
2019-03-09 02:16:38.547312: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:38.547312: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:38.547312: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:16:39 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:44 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 8900 lr 1 step-time 0.04s wps 67.49K ppl 1.04 gN 0.42 bleu 86.24, Sat Mar  9 02:16:47 2019
# Finished an epoch, step 8944. Perform external evaluation
2019-03-09 02:16:49.139540: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:49.139549: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:49.139573: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 1163
    src: which is taller between ranevskaya monument and aboriginal peoples in canada
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Aboriginal_peoples_in_Canada) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Atkinson_Clock_Tower) brack_close _oba_ var_b limit 1
2019-03-09 02:16:49.185830: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:49.185830: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:49.185830: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:16:49 2019.
  bleu dev: 86.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:16:55 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9000 lr 1 step-time 0.04s wps 65.75K ppl 1.04 gN 0.37 bleu 86.24, Sat Mar  9 02:16:59 2019
# Save eval, global step 9000
2019-03-09 02:16:59.521952: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:59.521994: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:59.521953: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 420
    src: give me the location of monument to michael the brave
    ref: select var_a where brack_open dbr_Monument_to_Michael_the_Brave,_Guruslău dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Michael_the_Brave,_Guruslău dbo_location var_a brack_close
2019-03-09 02:16:59.562145: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:16:59.562154: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:16:59.562156: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:16:59 2019.
  eval test: perplexity 1.05, time 2s, Sat Mar  9 02:17:01 2019.
# Finished an epoch, step 9030. Perform external evaluation
2019-03-09 02:17:02.893672: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:02.893672: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:02.893684: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 1081
    src: was dewey monument finished by <B>
    ref: ask where brack_open dbr_Dewey_Monument dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
    nmt: ask where brack_open dbr_Dewey_Monument dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
2019-03-09 02:17:02.930597: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:02.930597: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:02.930598: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:03 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:17:09 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9100 lr 1 step-time 0.03s wps 68.97K ppl 1.04 gN 0.38 bleu 86.46, Sat Mar  9 02:17:12 2019
# Finished an epoch, step 9116. Perform external evaluation
2019-03-09 02:17:13.524600: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:13.524601: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:13.524675: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 446
    src: where is mémorial des martyrs de la déportation located in
    ref: select var_a where brack_open dbr_Mémorial_des_Martyrs_de_la_Déportation dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Mémorial_des_Martyrs_de_la_Déportation dbo_location var_a brack_close
2019-03-09 02:17:13.561554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:13.561554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:13.561554: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:14 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:17:19 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9200 lr 1 step-time 0.03s wps 68.53K ppl 1.04 gN 0.39 bleu 86.46, Sat Mar  9 02:17:23 2019
# Finished an epoch, step 9202. Perform external evaluation
2019-03-09 02:17:23.985466: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:23.985491: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:23.985492: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 193
    src: mint clock tower
    ref: select var_a where brack_open dbr_Mint_Clock_Tower,_Chennai dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Mint_Clock_Tower,_Chennai dbo_abstract
2019-03-09 02:17:24.019469: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:24.019477: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:24.019477: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:24 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:17:30 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 9288. Perform external evaluation
2019-03-09 02:17:34.360300: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:34.360300: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:34.360300: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 999
    src: is bhasha smritistambha in india
    ref: ask where brack_open dbr_Bhasha_Smritistambha dbo_location dbr_India brack_close
    nmt: ask where brack_open dbr_Bhasha_Smritistambha dbo_location dbr_Kolkata brack_close
2019-03-09 02:17:34.394941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:34.394941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:34.394941: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:35 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:17:40 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9300 lr 1 step-time 0.04s wps 59.72K ppl 1.03 gN 0.39 bleu 86.46, Sat Mar  9 02:17:42 2019
# Finished an epoch, step 9374. Perform external evaluation
2019-03-09 02:17:44.569638: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:44.569657: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:44.569667: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 818
    src: latitude of prince henry the navigator
    ref: select var_a where brack_open dbr_Prince_Henry_the_Navigator_(statue) geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Prince_Henry_the_Navigator_(statue) geo_lat var_a brack_close
2019-03-09 02:17:44.604807: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:44.604808: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:44.604808: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:45 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:17:50 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9400 lr 1 step-time 0.03s wps 69.41K ppl 1.03 gN 0.36 bleu 86.46, Sat Mar  9 02:17:52 2019
# Finished an epoch, step 9460. Perform external evaluation
2019-03-09 02:17:54.765312: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:54.765326: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:54.765333: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 302
    src: where can one find confederate memorial
    ref: select var_a where brack_open dbr_Confederate_Memorial_(Wilmington,_North_Carolina) dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Confederate_Memorial_(Wilmington,_North_Carolina) dbo_location var_a brack_close
2019-03-09 02:17:54.801495: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:17:54.801509: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:17:54.801495: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:17:55 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:00 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9500 lr 1 step-time 0.03s wps 68.50K ppl 1.04 gN 0.39 bleu 86.46, Sat Mar  9 02:18:03 2019
# Finished an epoch, step 9546. Perform external evaluation
2019-03-09 02:18:05.243507: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:05.243510: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:05.243564: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 1065
    src: was garrick's temple to shakespeare finished by <B>
    ref: ask where brack_open dbr_Garrick's_Temple_to_Shakespeare dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
    nmt: ask where brack_open dbr_Garrick's_Temple_to_Shakespeare dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
2019-03-09 02:18:05.283832: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:05.283835: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:05.283837: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:05 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:11 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9600 lr 1 step-time 0.04s wps 68.25K ppl 1.03 gN 0.39 bleu 86.46, Sat Mar  9 02:18:14 2019
# Finished an epoch, step 9632. Perform external evaluation
2019-03-09 02:18:15.862081: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:15.862094: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:15.862081: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 855
    src: how north is newkirk viaduct monument
    ref: select var_a where brack_open dbr_Newkirk_Viaduct_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Newkirk_Viaduct_Monument geo_lat var_a brack_close
2019-03-09 02:18:15.900366: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:15.900366: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:15.900395: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:16 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:22 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9700 lr 1 step-time 0.03s wps 67.79K ppl 1.03 gN 0.36 bleu 86.46, Sat Mar  9 02:18:25 2019
# Finished an epoch, step 9718. Perform external evaluation
2019-03-09 02:18:26.467434: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:26.467434: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:26.467434: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 148
    src: russia–georgia friendship monument
    ref: select var_a where brack_open dbr_Russia–Georgia_Friendship_Monument dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Russia–Georgia_Friendship_Monument dbo_abstract
2019-03-09 02:18:26.502715: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:26.502719: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:26.502719: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:27 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:32 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9800 lr 1 step-time 0.03s wps 69.04K ppl 1.03 gN 0.38 bleu 86.46, Sat Mar  9 02:18:36 2019
# Finished an epoch, step 9804. Perform external evaluation
2019-03-09 02:18:36.844267: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:36.844267: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:36.844277: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 158
    src: monument of liberty
    ref: select var_a where brack_open dbr_Monument_of_Liberty,_Chișinău dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_of_Liberty,_Ruse dbo_abstract
2019-03-09 02:18:36.879768: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:36.879771: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:36.879777: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:37 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:43 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 9890. Perform external evaluation
2019-03-09 02:18:47.771097: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:47.771103: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:47.771106: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 831
    src: latitude of khachin-turbatli mausoleum
    ref: select var_a where brack_open dbr_Khachin-Turbatli_Mausoleum geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Khachin-Turbatli_Mausoleum geo_lat var_a brack_close
2019-03-09 02:18:47.807977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:47.807977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:47.807977: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:48 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:18:54 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 9900 lr 1 step-time 0.04s wps 57.79K ppl 1.03 gN 0.40 bleu 86.46, Sat Mar  9 02:18:56 2019
# Finished an epoch, step 9976. Perform external evaluation
2019-03-09 02:18:58.367011: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:58.367015: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:58.367019: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 71
    src: is romanian people's salvation cross a place
    ref: ask where brack_open dbr_Romanian_People's_Salvation_Cross rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Romanian_People's_Salvation_Cross rdf_type dbo_Place brack_close
2019-03-09 02:18:58.405321: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:18:58.405326: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:18:58.405326: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:18:59 2019.
  bleu dev: 86.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:04 2019.
  bleu test: 96.0
  saving hparams to /tmp/nmt_model/hparams
  step 10000 lr 1 step-time 0.03s wps 68.53K ppl 1.03 gN 0.40 bleu 86.46, Sat Mar  9 02:19:07 2019
# Save eval, global step 10000
2019-03-09 02:19:07.327477: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:07.327477: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:07.327495: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 615
    src: what is permyak salty ears related to
    ref: select var_a where brack_open dbr_Permyak_Salty_Ears dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Permyak_Salty_Ears dct_subject var_a brack_close
2019-03-09 02:19:07.364793: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:07.364794: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:07.364794: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:19:07 2019.
  eval test: perplexity 1.05, time 1s, Sat Mar  9 02:19:09 2019.
2019-03-09 02:19:09.579581: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:09.579595: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:09.579581: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 794
    src: latitude of livesey hall war memorial
    ref: select var_a where brack_open dbr_Livesey_Hall_War_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Livesey_Hall_War_Memorial geo_lat var_a brack_close
2019-03-09 02:19:09.615436: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:09.615436: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:09.615436: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:19:10 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:15 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 10062. Perform external evaluation
2019-03-09 02:19:19.028557: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:19.028557: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:19.028561: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 559
    src: what's monument to the liberator soldier native name
    ref: select var_a where brack_open dbr_Monument_to_the_Liberator_Soldier_(Kharkiv) dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_the_Liberator_Soldier_(Kharkiv) dbp_nativeName var_a brack_close
2019-03-09 02:19:19.066446: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:19.066446: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:19.066446: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:19:19 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:25 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10100 lr 1 step-time 0.03s wps 71.06K ppl 1.03 gN 0.41 bleu 86.56, Sat Mar  9 02:19:27 2019
# Finished an epoch, step 10148. Perform external evaluation
2019-03-09 02:19:29.400908: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:29.400908: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:29.400925: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 728
    src: what is monument to salavat yulaev about
    ref: select var_a where brack_open dbr_Monument_to_Salavat_Yulaev dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Salavat_Yulaev dct_subject var_a brack_close
2019-03-09 02:19:29.436615: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:29.436619: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:29.436619: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:19:30 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:35 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10200 lr 1 step-time 0.03s wps 68.93K ppl 1.03 gN 0.37 bleu 86.56, Sat Mar  9 02:19:38 2019
# Finished an epoch, step 10234. Perform external evaluation
2019-03-09 02:19:39.910379: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:39.910390: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:39.910400: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 549
    src: what's wat's dyke native name
    ref: select var_a where brack_open dbr_Wat's_Dyke dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Offa's_Dyke dbp_nativeName var_a brack_close
2019-03-09 02:19:39.948873: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:39.948873: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:39.948880: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:19:40 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:46 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10300 lr 1 step-time 0.03s wps 68.63K ppl 1.03 gN 0.39 bleu 86.56, Sat Mar  9 02:19:49 2019
# Finished an epoch, step 10320. Perform external evaluation
2019-03-09 02:19:50.404165: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:50.404165: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:50.404222: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 979
    src: when was perry monument built
    ref: select var_a where brack_open dbr_Perry_Monument_(Cleveland) dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Perry_Monument_(Cleveland) dbp_complete var_a brack_close
2019-03-09 02:19:50.440669: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:19:50.440669: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:19:50.440671: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:19:51 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:19:56 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10400 lr 1 step-time 0.04s wps 68.27K ppl 1.03 gN 0.36 bleu 86.56, Sat Mar  9 02:20:00 2019
# Finished an epoch, step 10406. Perform external evaluation
2019-03-09 02:20:00.721229: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:00.721248: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:00.721250: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 935
    src: longitude of grand bazaar
    ref: select var_a where brack_open dbr_Grand_Bazaar,_Tehran geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Grand_Bazaar,_Tehran geo_long var_a brack_close
2019-03-09 02:20:00.757633: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:00.757637: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:00.757637: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:01 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:20:06 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 10492. Perform external evaluation
2019-03-09 02:20:11.014356: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:11.014356: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:11.014376: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 736
    src: what are the coordinates of dewey monument
    ref: select var_a where brack_open dbr_Dewey_Monument georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Dewey_Monument georss_point var_a brack_close
2019-03-09 02:20:11.050178: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:11.050192: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:11.050178: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:11 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:20:17 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10500 lr 1 step-time 0.04s wps 59.67K ppl 1.03 gN 0.39 bleu 86.56, Sat Mar  9 02:20:19 2019
# Finished an epoch, step 10578. Perform external evaluation
2019-03-09 02:20:21.532894: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:21.532909: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:21.532905: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 930
    src: longitude of alley of classics
    ref: select var_a where brack_open dbr_Alley_of_Classics,_Chișinău geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Alley_of_Classics,_Bălți geo_long var_a brack_close
2019-03-09 02:20:21.568232: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:21.568232: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:21.568232: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:22 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:20:27 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10600 lr 1 step-time 0.03s wps 68.24K ppl 1.03 gN 0.34 bleu 86.56, Sat Mar  9 02:20:29 2019
# Finished an epoch, step 10664. Perform external evaluation
2019-03-09 02:20:31.544626: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:31.544639: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:31.544626: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 294
    src: where can one find husainabad clock tower
    ref: select var_a where brack_open dbr_Husainabad_Clock_Tower dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Husainabad_Clock_Tower dbo_location var_a brack_close
2019-03-09 02:20:31.580447: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:31.580447: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:31.580450: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:32 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:20:37 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10700 lr 1 step-time 0.03s wps 69.59K ppl 1.03 gN 0.36 bleu 86.56, Sat Mar  9 02:20:40 2019
# Finished an epoch, step 10750. Perform external evaluation
2019-03-09 02:20:41.542694: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:41.542710: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:41.542694: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 244
    src: where is dupont circle fountain
    ref: select var_a where brack_open dbr_Dupont_Circle_Fountain dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Dupont_Circle_Fountain dbo_location var_a brack_close
2019-03-09 02:20:41.578641: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:41.578641: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:41.578641: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:42 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:20:47 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10800 lr 1 step-time 0.03s wps 69.30K ppl 1.02 gN 0.33 bleu 86.56, Sat Mar  9 02:20:50 2019
# Finished an epoch, step 10836. Perform external evaluation
2019-03-09 02:20:51.660804: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:51.660804: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:51.660812: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 297
    src: where can one find monument aux braves de n.d.g.
    ref: select var_a where brack_open dbr_Monument_aux_braves_de_N.D.G sep_dot dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_aux_braves_de_N.D.G sep_dot dbo_location var_a brack_close
2019-03-09 02:20:51.699067: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:20:51.699067: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:20:51.699073: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:20:52 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:20:57 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 10900 lr 1 step-time 0.03s wps 68.70K ppl 1.03 gN 0.34 bleu 86.56, Sat Mar  9 02:21:01 2019
# Finished an epoch, step 10922. Perform external evaluation
2019-03-09 02:21:01.983914: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:01.983915: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:01.983976: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 943
    src: building date of monument to nizami ganjavi in tashkent
    ref: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Tashkent dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Tashkent dbp_complete var_a brack_close
2019-03-09 02:21:02.019730: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:02.019731: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:02.019730: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:02 2019.
  bleu dev: 86.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Sat Mar  9 02:21:07 2019.
  bleu test: 96.1
  saving hparams to /tmp/nmt_model/hparams
  step 11000 lr 1 step-time 0.04s wps 66.45K ppl 1.02 gN 0.34 bleu 86.56, Sat Mar  9 02:21:11 2019
# Save eval, global step 11000
2019-03-09 02:21:12.136666: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:12.136688: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:12.136668: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 1273
    src: give me the transport in europe tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Transport_in_Europe
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Narasimhaswamy_Temple,_Namakkal
2019-03-09 02:21:12.180607: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:12.180607: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:12.180607: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:21:12 2019.
  eval test: perplexity 1.04, time 2s, Sat Mar  9 02:21:14 2019.
# Finished an epoch, step 11008. Perform external evaluation
2019-03-09 02:21:14.911889: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:14.911905: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:14.911905: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 1218
    src: which is longer marco zero or ramone mckenzie
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Marco_Zero_(São_Paulo) || var_a = dbr_Ramone_McKenzie) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Marco_Zero_(São_Paulo) || var_a = dbr_New_Netherland) brack_close
2019-03-09 02:21:14.952128: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:14.952130: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:14.952130: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:15 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:21:21 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 11094. Perform external evaluation
2019-03-09 02:21:25.684179: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:25.684187: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:25.684221: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 156
    src: capitoline wolf
    ref: select var_a where brack_open dbr_Capitoline_Wolf,_Chișinău dbo_abstract var_a brack_close
    nmt: select var_a where brack_open
2019-03-09 02:21:25.719512: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:25.719512: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:25.719512: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:26 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:21:31 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11100 lr 1 step-time 0.04s wps 58.63K ppl 1.03 gN 0.49 bleu 87.31, Sat Mar  9 02:21:33 2019
# Finished an epoch, step 11180. Perform external evaluation
2019-03-09 02:21:35.930378: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:35.930382: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:35.930401: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 456
    src: where is commando memorial located in
    ref: select var_a where brack_open dbr_Commando_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Commando_Memorial dbo_location var_a brack_close
2019-03-09 02:21:35.968028: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:35.968028: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:35.968030: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:36 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:21:41 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11200 lr 1 step-time 0.03s wps 67.23K ppl 1.02 gN 0.36 bleu 87.31, Sat Mar  9 02:21:44 2019
# Finished an epoch, step 11266. Perform external evaluation
2019-03-09 02:21:46.113047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:46.113047: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:46.113054: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 732
    src: what is llama de la libertad about
    ref: select var_a where brack_open dbr_Llama_de_la_Libertad dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Llama_de_la_Libertad dct_subject var_a brack_close
2019-03-09 02:21:46.149967: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:46.149967: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:46.149967: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:46 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:21:52 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11300 lr 1 step-time 0.03s wps 69.64K ppl 1.02 gN 0.37 bleu 87.31, Sat Mar  9 02:21:54 2019
# Finished an epoch, step 11352. Perform external evaluation
2019-03-09 02:21:56.538603: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:56.538613: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:56.538615: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 998
    src: is yalova earthquake monument in yalova
    ref: ask where brack_open dbr_Yalova_Earthquake_Monument dbo_location dbr_Yalova brack_close
    nmt: ask where brack_open dbr_Yalova_Earthquake_Monument dbo_location dbr_Beaufort,_Malaysia brack_close
2019-03-09 02:21:56.578408: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:21:56.578408: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:21:56.578416: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:21:57 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:02 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11400 lr 1 step-time 0.04s wps 66.82K ppl 1.02 gN 0.36 bleu 87.31, Sat Mar  9 02:22:05 2019
# Finished an epoch, step 11438. Perform external evaluation
2019-03-09 02:22:07.213129: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:07.213129: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:07.213174: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 456
    src: where is commando memorial located in
    ref: select var_a where brack_open dbr_Commando_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Commando_Memorial dbo_location var_a brack_close
2019-03-09 02:22:07.252339: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:07.252339: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:07.252341: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:22:07 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:13 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11500 lr 1 step-time 0.03s wps 68.43K ppl 1.02 gN 0.32 bleu 87.31, Sat Mar  9 02:22:17 2019
# Finished an epoch, step 11524. Perform external evaluation
2019-03-09 02:22:17.953628: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:17.953647: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:17.953632: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 754
    src: what are the coordinates of sun yat-sen memorial hall
    ref: select var_a where brack_open dbr_Sun_Yat-sen_Memorial_Hall_(Guangzhou) georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Sun_Yat-sen_Memorial_Hall_(Taipei) georss_point var_a brack_close
2019-03-09 02:22:17.990173: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:17.990173: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:17.990173: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:22:18 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:23 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11600 lr 1 step-time 0.04s wps 69.13K ppl 1.02 gN 0.32 bleu 87.31, Sat Mar  9 02:22:27 2019
# Finished an epoch, step 11610. Perform external evaluation
2019-03-09 02:22:28.277446: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:28.277450: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:28.277473: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 583
    src: what is monument of liberty related to
    ref: select var_a where brack_open dbr_Monument_of_Liberty,_Chișinău dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_of_Liberty,_Ruse dct_subject var_a brack_close
2019-03-09 02:22:28.316046: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:28.316046: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:28.316046: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:22:29 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:34 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 11696. Perform external evaluation
2019-03-09 02:22:38.910809: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:38.910820: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:38.910835: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 682
    src: what is monarch advertising sign about
    ref: select var_a where brack_open dbr_Monarch_advertising_sign dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monarch_advertising_sign dct_subject var_a brack_close
2019-03-09 02:22:38.947291: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:38.947291: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:38.947291: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:22:39 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:45 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11700 lr 1 step-time 0.04s wps 59.84K ppl 1.02 gN 0.35 bleu 87.31, Sat Mar  9 02:22:46 2019
# Finished an epoch, step 11782. Perform external evaluation
2019-03-09 02:22:49.382549: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:49.382549: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:49.382559: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 18
    src: is reformation memorial a monument
    ref: ask where brack_open dbr_Reformation_Memorial,_Copenhagen rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Reformation_Memorial,_Copenhagen rdf_type dbo_Monument brack_close
2019-03-09 02:22:49.418214: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:49.418214: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:49.418217: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:22:50 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:22:55 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11800 lr 1 step-time 0.04s wps 65.02K ppl 1.02 gN 0.34 bleu 87.31, Sat Mar  9 02:22:57 2019
# Finished an epoch, step 11868. Perform external evaluation
2019-03-09 02:22:59.910838: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:59.910867: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:59.910867: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 1312
    src: what do our lady of the rockies and our lady of the rockies have in common
    ref: select wildcard where brack_open brack_open dbr_Our_Lady_of_the_Rockies var_a var_b sep_dot dbr_Our_Lady_of_the_Rockies var_a var_b brack_close UNION brack_open brack_open dbr_Our_Lady_of_the_Rockies var_a var_b sep_dot dbr_Our_Lady_of_the_Rockies var_a var_b brack_close UNION brack_open var_c var_d dbr_Our_Lady_of_the_Rockies sep_dot var_c var_d dbr_Our_Lady_of_the_Rockies brack_close brack_close UNION brack_open var_c var_d dbr_Our_Lady_of_the_Rockies sep_dot var_c var_d dbr_Our_Lady_of_the_Rockies brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Our_Lady_of_the_Rockies var_a var_b sep_dot dbr_Catacombs_of_Kom_el_Shoqafa var_a var_b brack_close UNION brack_open brack_open dbr_Catacombs_of_Kom_el_Shoqafa var_a var_b sep_dot dbr_Catacombs_of_Kom_el_Shoqafa var_a var_b brack_close UNION brack_open var_c var_d dbr_Catacombs_of_Kom_el_Shoqafa sep_dot var_c var_d
2019-03-09 02:22:59.966848: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:22:59.966848: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:22:59.966848: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:23:00 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:23:06 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 11900 lr 1 step-time 0.04s wps 66.54K ppl 1.02 gN 0.36 bleu 87.31, Sat Mar  9 02:23:08 2019
# Finished an epoch, step 11954. Perform external evaluation
2019-03-09 02:23:10.512776: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:10.512776: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:10.512811: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 72
    src: is monumento de santiago a monument
    ref: ask where brack_open dbr_Monumento_de_Santiago rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Monumento_de_Santiago rdf_type dbo_Monument brack_close
2019-03-09 02:23:10.549328: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:10.549329: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:10.549329: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:23:11 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:23:16 2019.
  bleu test: 96.3
  saving hparams to /tmp/nmt_model/hparams
  step 12000 lr 1 step-time 0.03s wps 68.23K ppl 1.02 gN 0.33 bleu 87.31, Sat Mar  9 02:23:19 2019
# Save eval, global step 12000
2019-03-09 02:23:19.815418: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:19.815422: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:19.815427: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  # 1132
    src: which is taller between newkirk viaduct monument and bowie seamount
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Bowie_Seamount) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Santa_Quiteria_Bridge) brack_close _oba_ var_b limit 1
2019-03-09 02:23:19.860048: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:19.860048: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:19.860051: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:23:20 2019.
  eval test: perplexity 1.04, time 1s, Sat Mar  9 02:23:21 2019.
2019-03-09 02:23:21.975487: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:21.975504: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:21.975498: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  # 1300
    src: what do sun yat-sen memorial hall and sun yat-sen memorial hall have in common
    ref: select wildcard where brack_open brack_open dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close var_a var_b sep_dot dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close var_a var_b brack_close UNION brack_open brack_open dbr_Sun_Yat-sen_Memorial_Hall_(Guangzhou) var_a var_b sep_dot dbr_Sun_Yat-sen_Memorial_Hall_(Guangzhou) var_a var_b brack_close UNION brack_open var_c var_d dbr_Sun_Yat-sen_Memorial_Hall_(Guangzhou) sep_dot var_c var_d dbr_Sun_Yat-sen_Memorial_Hall_(Guangzhou) brack_close brack_close UNION brack_open var_c var_d dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close sep_dot var_c var_d dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close var_a var_b sep_dot dbr_Sun_Yat-sen_Memorial_Hall_ attr_open Guangzhou attr_close var_a var_b brack_close UNION brack_open brack_open dbr_Sun_Yat-sen_Memorial_Hall_(Taipei) var_a var_b sep_dot dbr_Sun_Yat-sen_Memorial_Hall_(Taipei) var_a
2019-03-09 02:23:22.027129: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:22.027150: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:22.027150: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.03s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:23:22 2019.
  eval test: perplexity 1.04, time 1s, Sat Mar  9 02:23:24 2019.
2019-03-09 02:23:24.266621: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:24.266621: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:24.266627: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.03s
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:23:24 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:23:30 2019.
  bleu test: 96.6
  saving hparams to /tmp/nmt_model/hparams
# Final, step 12000 lr 1 step-time 0.03s wps 68.23K ppl 1.02 gN 0.33 dev ppl 1.26, dev bleu 87.3, test ppl 1.04, test bleu 96.6, Sat Mar  9 02:23:31 2019
# Done training!, time 1496s, Sat Mar  9 02:23:31 2019.
# Start evaluating saved best models.
2019-03-09 02:23:31.660897: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:31.660897: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:31.660898: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
  loaded infer model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.03s
  # 307
    src: where can one find pelican pete
    ref: select var_a where brack_open dbr_Pelican_Pete dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Pelican_Pete dbo_location var_a brack_close
2019-03-09 02:23:31.696062: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:31.696062: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:31.696067: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded eval model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.02s
  eval dev: perplexity 1.26, time 0s, Sat Mar  9 02:23:31 2019.
  eval test: perplexity 1.04, time 1s, Sat Mar  9 02:23:33 2019.
2019-03-09 02:23:33.759082: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.en is already initialized.
2019-03-09 02:23:33.759088: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
2019-03-09 02:23:33.759088: I tensorflow/core/kernels/lookup_util.cc:373] Table trying to initialize from file /tmp/nmt_model/vocab.sparql is already initialized.
  loaded infer model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.03s
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Sat Mar  9 02:23:34 2019.
  bleu dev: 87.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Sat Mar  9 02:23:39 2019.
  bleu test: 96.6
  saving hparams to /tmp/nmt_model/hparams
# Best bleu, step 12000 lr 1 step-time 0.03s wps 68.23K ppl 1.02 gN 0.33 dev ppl 1.26, dev bleu 87.3, test ppl 1.04, test bleu 96.6, Sat Mar  9 02:23:41 2019
