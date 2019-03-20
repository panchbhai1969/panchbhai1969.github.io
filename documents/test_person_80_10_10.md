# Job id 0
# Devices visible to TensorFlow: [_DeviceAttributes(/job:localhost/replica:0/task:0/device:CPU:0, CPU, 268435456, 81710344362223773), _DeviceAttributes(/job:localhost/replica:0/task:0/device:GPU:0, GPU, 3126984704, 10244432681569383448)]
# Loading hparams from /tmp/nmt_model/hparams
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
  best_bleu=24.9965909325
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
  metrics=[u'bleu']
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
# log_file=/tmp/nmt_model/log_1553117283
  loaded train model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.08s
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.03s
  # 753
    src: what are the coordinates of raymond's tomb
    ref: select var_a where brack_open dbr_Raymond's_Tomb georss_point var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.03s
  eval dev: perplexity 2.48, time 0s, Thu Mar 21 02:58:04 2019.
  eval test: perplexity 2.49, time 1s, Thu Mar 21 02:58:05 2019.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:58:06 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 02:58:12 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
# Start step 1000, lr 1, Thu Mar 21 02:58:13 2019
# Init train iterator, skipping 0 elements
# Finished an epoch, step 1086. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 939
    src: building date of maison coilliot
    ref: select var_a where brack_open dbr_Maison_Coilliot dbp_complete var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:58:18 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:58:24 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1100 lr 1 step-time 0.04s wps 53.95K ppl 2.33 gN 1.25 bleu 25.00, Thu Mar 21 02:58:26 2019
# Finished an epoch, step 1172. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1170
    src: which is longer ahu akivi or giant-impact hypothesis
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Ahu_Akivi || var_a = dbr_Giant-impact_hypothesis) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_February_One || var_a = dbr_Fiona_Corke) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:58:29 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:58:35 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1200 lr 1 step-time 0.03s wps 76.43K ppl 2.30 gN 1.24 bleu 25.00, Thu Mar 21 02:58:38 2019
# Finished an epoch, step 1258. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1034
    src: is major general john a. logan in logan circle
    ref: ask where brack_open dbr_Major_General_John_A._Logan dbo_location dbr_Logan_Circle,_Washington,_D.C sep_dot brack_close
    nmt: ask where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_February_One || var_a = dbr_Fiona_Corke) brack_close _oba_ var_b limit
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:58:40 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:58:46 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1300 lr 1 step-time 0.03s wps 73.85K ppl 2.19 gN 1.41 bleu 25.00, Thu Mar 21 02:58:49 2019
# Finished an epoch, step 1344. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 362
    src: location of monument to nicholas i
    ref: select var_a where brack_open dbr_Monument_to_Nicholas_I dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:58:51 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:58:58 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1400 lr 1 step-time 0.03s wps 75.51K ppl 2.23 gN 0.94 bleu 25.00, Thu Mar 21 02:59:01 2019
# Finished an epoch, step 1430. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 1267
    src: give me the thillenkeri tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Thillenkeri
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:59:02 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:59:09 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1500 lr 1 step-time 0.03s wps 74.26K ppl 2.15 gN 0.92 bleu 25.00, Thu Mar 21 02:59:12 2019
# Finished an epoch, step 1516. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 822
    src: latitude of respect to mehmetçik monument
    ref: select var_a where brack_open dbr_Respect_to_Mehmetçik_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:59:14 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:59:20 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1600 lr 1 step-time 0.03s wps 74.28K ppl 2.13 gN 0.95 bleu 25.00, Thu Mar 21 02:59:24 2019
# Finished an epoch, step 1602. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 31
    src: is buzeyir cave a cave
    ref: ask where brack_open dbr_Buzeyir_cave rdf_type dbo_Cave brack_close
    nmt: ask where brack_open dbr_Garghabazar_Mosque rdf_type dbo_Monument brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:59:25 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:59:32 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 1688. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 301
    src: where can one find ignace bourget monument
    ref: select var_a where brack_open dbr_Ignace_Bourget_Monument dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:59:37 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:59:43 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1700 lr 1 step-time 0.04s wps 63.25K ppl 2.14 gN 0.99 bleu 25.00, Thu Mar 21 02:59:45 2019
# Finished an epoch, step 1774. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 421
    src: give me the location of temple of augustus
    ref: select var_a where brack_open dbr_Temple_of_Augustus,_Pula dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit limit
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 02:59:48 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 02:59:55 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1800 lr 1 step-time 0.03s wps 74.05K ppl 2.07 gN 0.84 bleu 25.00, Thu Mar 21 02:59:57 2019
# Finished an epoch, step 1860. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 326
    src: location of general john a. rawlins
    ref: select var_a where brack_open dbr_General_John_A._Rawlins dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:00:00 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:00:07 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 1900 lr 1 step-time 0.03s wps 73.42K ppl 2.01 gN 1.03 bleu 25.00, Thu Mar 21 03:00:10 2019
# Finished an epoch, step 1946. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
  # 340
    src: location of catacombs of kom el shoqafa
    ref: select var_a where brack_open dbr_Catacombs_of_Kom_el_Shoqafa dbo_location var_a brack_close
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-1000, time 0.02s
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:00:12 2019.
  bleu dev: 25.0
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 1000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:00:18 2019.
  bleu test: 24.9
  saving hparams to /tmp/nmt_model/hparams
  step 2000 lr 1 step-time 0.03s wps 73.95K ppl 2.02 gN 1.18 bleu 25.00, Thu Mar 21 03:00:22 2019
# Save eval, global step 2000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 652
    src: what is cenotaph all about
    ref: select var_a where brack_open dbr_Cenotaph_(Montreal) dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Vojinović_Bridge dct_subject var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  eval dev: perplexity 2.15, time 0s, Thu Mar 21 03:00:22 2019.
  eval test: perplexity 2.03, time 1s, Thu Mar 21 03:00:24 2019.
# Finished an epoch, step 2032. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 50
    src: is castillo de san andrés a place
    ref: ask where brack_open dbr_Castillo_de_San_Andrés rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Monumental_Clock_of_Pachuca rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:00:25 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:00:30 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2100 lr 1 step-time 0.03s wps 72.53K ppl 1.98 gN 0.97 bleu 58.46, Thu Mar 21 03:00:34 2019
# Finished an epoch, step 2118. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 446
    src: where is mémorial des martyrs de la déportation located in
    ref: select var_a where brack_open dbr_Mémorial_des_Martyrs_de_la_Déportation dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Canadian_Tomb_of_the_Unknown_Soldier dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:00:35 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 11s, Thu Mar 21 03:00:47 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2200 lr 1 step-time 0.05s wps 43.67K ppl 1.97 gN 0.87 bleu 58.46, Thu Mar 21 03:00:54 2019
# Finished an epoch, step 2204. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.03s
  # 1039
    src: is monumento a la abolición de la esclavitud in ponce
    ref: ask where brack_open dbr_Monumento_a_la_abolición_de_la_esclavitud dbo_location dbr_Ponce,_Puerto_Rico brack_close
    nmt: ask where brack_open dbr_Canadian_Tomb_of_the_Unknown_Soldier dbo_location dbr_Azerbaijan brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.03s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:00:55 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 8s, Thu Mar 21 03:01:04 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 2290. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1317
    src: what do castle of capdepera and castle of capdepera have in common
    ref: select wildcard where brack_open brack_open dbr_Castle_of_Capdepera var_a var_b sep_dot dbr_Castle_of_Capdepera var_a var_b brack_close UNION brack_open brack_open dbr_Castle_of_Capdepera var_a var_b sep_dot dbr_Castle_of_Capdepera var_a var_b brack_close UNION brack_open var_c var_d dbr_Castle_of_Capdepera sep_dot var_c var_d dbr_Castle_of_Capdepera brack_close brack_close UNION brack_open var_c var_d dbr_Castle_of_Capdepera sep_dot var_c var_d dbr_Castle_of_Capdepera brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Nagasaki_Peace_Park var_a var_b sep_dot dbr_Eternal_flame var_a var_b brack_close UNION brack_open brack_open dbr_Bonifacio_Monument var_a var_b sep_dot dbr_Eternal_flame var_a var_b brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:01:09 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:01:15 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2300 lr 1 step-time 0.05s wps 49.27K ppl 1.99 gN 1.23 bleu 58.46, Thu Mar 21 03:01:17 2019
# Finished an epoch, step 2376. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 904
    src: longitude of caverne du pont-d'arc
    ref: select var_a where brack_open dbr_Caverne_du_Pont-d'Arc geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Kurunegala_Clock_Tower geo_long var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:01:21 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:01:26 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2400 lr 1 step-time 0.04s wps 67.56K ppl 1.92 gN 0.87 bleu 58.46, Thu Mar 21 03:01:28 2019
# Finished an epoch, step 2462. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 297
    src: where can one find monument aux braves de n.d.g.
    ref: select var_a where brack_open dbr_Monument_aux_braves_de_N.D.G sep_dot dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Memorial_for_the_victims_killed_by_OUN-UPA_(Luhansk) dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:01:31 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:01:36 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2500 lr 1 step-time 0.03s wps 71.64K ppl 1.84 gN 0.83 bleu 58.46, Thu Mar 21 03:01:39 2019
# Finished an epoch, step 2548. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 72
    src: is monumento de santiago a monument
    ref: ask where brack_open dbr_Monumento_de_Santiago rdf_type dbo_Monument brack_close
    nmt: ask where brack_open dbr_Monumental_Clock_of_Pachuca rdf_type dbo_Monument brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:01:41 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:01:46 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2600 lr 1 step-time 0.03s wps 73.79K ppl 1.85 gN 1.00 bleu 58.46, Thu Mar 21 03:01:49 2019
# Finished an epoch, step 2634. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 470
    src: where is jesus de greatest located in
    ref: select var_a where brack_open dbr_Jesus_de_Greatest dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Temple_of_Augustus,_Pula dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:01:51 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:01:56 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2700 lr 1 step-time 0.03s wps 75.40K ppl 1.76 gN 0.92 bleu 58.46, Thu Mar 21 03:01:59 2019
# Finished an epoch, step 2720. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 935
    src: longitude of grand bazaar
    ref: select var_a where brack_open dbr_Grand_Bazaar,_Tehran geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Pegasus_and_Dragon geo_long var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:00 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:02:05 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2800 lr 1 step-time 0.03s wps 76.44K ppl 1.78 gN 1.06 bleu 58.46, Thu Mar 21 03:02:09 2019
# Finished an epoch, step 2806. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 622
    src: what is confederate memorial park related to
    ref: select var_a where brack_open dbr_Confederate_Memorial_Park_(Albany,_Georgia) dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Transfiguration_Cathedral_in_Odessa dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:09 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:02:15 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 2892. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1020
    src: is kandy clock tower in kandy
    ref: ask where brack_open dbr_Kandy_Clock_Tower dbo_location dbr_Kandy brack_close
    nmt: ask where brack_open dbr_La_Barre_Monument dbo_location dbr_Azerbaijan brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:19 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:02:24 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 2900 lr 1 step-time 0.04s wps 64.77K ppl 1.77 gN 1.34 bleu 58.46, Thu Mar 21 03:02:26 2019
# Finished an epoch, step 2978. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
  # 1298
    src: what do ludlow monument and ludlow monument have in common
    ref: select wildcard where brack_open brack_open dbr_Ludlow_Monument var_a var_b sep_dot dbr_Ludlow_Monument var_a var_b brack_close UNION brack_open brack_open dbr_Ludlow_Monument var_a var_b sep_dot dbr_Ludlow_Monument var_a var_b brack_close UNION brack_open var_c var_d dbr_Ludlow_Monument sep_dot var_c var_d dbr_Ludlow_Monument brack_close brack_close UNION brack_open var_c var_d dbr_Ludlow_Monument sep_dot var_c var_d dbr_Ludlow_Monument brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Nagasaki_Peace_Park var_a var_b sep_dot dbr_Eternal_flame var_a var_b brack_close UNION brack_open brack_open dbr_Bonifacio_Monument var_a var_b sep_dot
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-2000, time 0.02s
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:29 2019.
  bleu dev: 58.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 2000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:02:34 2019.
  bleu test: 59.1
  saving hparams to /tmp/nmt_model/hparams
  step 3000 lr 1 step-time 0.03s wps 73.81K ppl 1.68 gN 1.16 bleu 58.46, Thu Mar 21 03:02:36 2019
# Save eval, global step 3000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 244
    src: where is dupont circle fountain
    ref: select var_a where brack_open dbr_Dupont_Circle_Fountain dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Taras_Shevchenko_Memorial dbo_location var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  eval dev: perplexity 1.83, time 0s, Thu Mar 21 03:02:36 2019.
  eval test: perplexity 1.65, time 1s, Thu Mar 21 03:02:38 2019.
# Finished an epoch, step 3064. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1277
    src: what do chaitya bhoomi and chaitya bhoomi have in common
    ref: select wildcard where brack_open brack_open dbr_Chaitya_Bhoomi var_a var_b sep_dot dbr_Chaitya_Bhoomi var_a var_b brack_close UNION brack_open brack_open dbr_Chaitya_Bhoomi var_a var_b sep_dot dbr_Chaitya_Bhoomi var_a var_b brack_close UNION brack_open var_c var_d dbr_Chaitya_Bhoomi sep_dot var_c var_d dbr_Chaitya_Bhoomi brack_close brack_close UNION brack_open var_c var_d dbr_Chaitya_Bhoomi sep_dot var_c var_d dbr_Chaitya_Bhoomi brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Anıtkabir attr_open Montreal attr_close var_a var_b sep_dot dbr_South_African_War_Memorial_ attr_open Montreal attr_close var_a var_b brack_close UNION
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:40 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:02:45 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3100 lr 1 step-time 0.03s wps 73.41K ppl 1.66 gN 1.13 bleu 62.41, Thu Mar 21 03:02:48 2019
# Finished an epoch, step 3150. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 1032
    src: is clock tower in faisalabad
    ref: ask where brack_open dbr_Clock_Tower,_Faisalabad dbo_location dbr_Faisalabad brack_close
    nmt: ask where brack_open dbr_Clock_Tower,_Faisalabad dbo_location dbr_Azerbaijan brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:02:50 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:02:56 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3200 lr 1 step-time 0.03s wps 71.67K ppl 1.63 gN 1.35 bleu 62.41, Thu Mar 21 03:02:58 2019
# Finished an epoch, step 3236. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 627
    src: what is countess pillar related to
    ref: select var_a where brack_open dbr_Countess_Pillar dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Abraham_Lincoln_(Flannery) dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:00 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:03:05 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3300 lr 1 step-time 0.03s wps 72.53K ppl 1.56 gN 1.29 bleu 62.41, Thu Mar 21 03:03:09 2019
# Finished an epoch, step 3322. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 430
    src: where is february one located in
    ref: select var_a where brack_open dbr_February_One dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_February_One dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:10 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:03:15 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3400 lr 1 step-time 0.03s wps 72.25K ppl 1.54 gN 1.28 bleu 62.41, Thu Mar 21 03:03:19 2019
# Finished an epoch, step 3408. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 773
    src: what are the coordinates of uhuru monument
    ref: select var_a where brack_open dbr_Uhuru_Monument georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Washington_Monument_(Baltimore) georss_point var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:20 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:03:24 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 3494. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 876
    src: how north is alabama state monument
    ref: select var_a where brack_open dbr_Alabama_State_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Adam_Mickiewicz_Monument,_Warsaw geo_lat var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:29 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:03:34 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3500 lr 1 step-time 0.04s wps 61.39K ppl 1.51 gN 1.51 bleu 62.41, Thu Mar 21 03:03:36 2019
# Finished an epoch, step 3580. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 494
    src: who designed old city hall cenotaph
    ref: select var_a where brack_open dbr_Old_City_Hall_Cenotaph,_Toronto dbo_designer var_a brack_close
    nmt: select var_a where brack_open dbr_Garrick's_Temple_to_Shakespeare dbo_designer var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:39 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:03:44 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3600 lr 1 step-time 0.03s wps 69.45K ppl 1.44 gN 1.26 bleu 62.41, Thu Mar 21 03:03:47 2019
# Finished an epoch, step 3666. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 324
    src: location of grotto of our lady of lourdes
    ref: select var_a where brack_open dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Our_Lady_of_the_Rockies dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:03:49 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:03:55 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3700 lr 1 step-time 0.04s wps 68.61K ppl 1.41 gN 1.24 bleu 62.41, Thu Mar 21 03:03:57 2019
# Finished an epoch, step 3752. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 360
    src: location of south african war memorial
    ref: select var_a where brack_open dbr_South_African_War_Memorial_(Toronto) dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Diana_the_Huntress_Fountain dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:00 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:04:05 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3800 lr 1 step-time 0.03s wps 70.97K ppl 1.37 gN 1.29 bleu 62.41, Thu Mar 21 03:04:08 2019
# Finished an epoch, step 3838. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 965
    src: when was tomb of payava built
    ref: select var_a where brack_open dbr_Tomb_of_Payava dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Statue_of_Honor dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:10 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:04:15 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 3900 lr 1 step-time 0.03s wps 70.10K ppl 1.35 gN 1.40 bleu 62.41, Thu Mar 21 03:04:19 2019
# Finished an epoch, step 3924. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
  # 345
    src: location of allahabad pillar
    ref: select var_a where brack_open dbr_Allahabad_pillar dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Urasoe_yōdore dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-3000, time 0.02s
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:20 2019.
  bleu dev: 62.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 3000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:04:26 2019.
  bleu test: 64.9
  saving hparams to /tmp/nmt_model/hparams
  step 4000 lr 1 step-time 0.03s wps 70.33K ppl 1.34 gN 1.62 bleu 62.41, Thu Mar 21 03:04:29 2019
# Save eval, global step 4000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 747
    src: what are the coordinates of garibaldi monument in taganrog
    ref: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog georss_point var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  eval dev: perplexity 1.52, time 0s, Thu Mar 21 03:04:30 2019.
  eval test: perplexity 1.31, time 1s, Thu Mar 21 03:04:32 2019.
# Finished an epoch, step 4010. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 328
    src: location of wilfrid laurier memorial
    ref: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:33 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:04:38 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 4096. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 660
    src: what is maha kavi moyinkutty vaidyar smaraka all about
    ref: select var_a where brack_open dbr_Maha_Kavi_Moyinkutty_Vaidyar_Smaraka dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Evil_Clown_of_Middletown dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:43 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:04:48 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4100 lr 1 step-time 0.04s wps 60.83K ppl 1.30 gN 1.44 bleu 74.32, Thu Mar 21 03:04:49 2019
# Finished an epoch, step 4182. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 212
    src: where is shot at dawn memorial
    ref: select var_a where brack_open dbr_Shot_at_Dawn_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Sverd_i_fjell dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:04:53 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:04:58 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4200 lr 1 step-time 0.04s wps 66.42K ppl 1.26 gN 1.17 bleu 74.32, Thu Mar 21 03:05:01 2019
# Finished an epoch, step 4268. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 57
    src: is abraham lincoln a place
    ref: ask where brack_open dbr_Abraham_Lincoln_(Flannery) rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Hagia_Sophia rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:03 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:05:09 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4300 lr 1 step-time 0.03s wps 71.22K ppl 1.25 gN 1.25 bleu 74.32, Thu Mar 21 03:05:11 2019
# Finished an epoch, step 4354. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 531
    src: when was shrine of the book completed
    ref: select var_a where brack_open dbr_Shrine_of_the_Book dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Shrine_of_the_Book dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:14 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:05:19 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4400 lr 1 step-time 0.03s wps 71.03K ppl 1.24 gN 1.45 bleu 74.32, Thu Mar 21 03:05:22 2019
# Finished an epoch, step 4440. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 279
    src: where can one find garibaldi monument in taganrog
    ref: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:24 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:05:29 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4500 lr 1 step-time 0.03s wps 70.94K ppl 1.21 gN 1.15 bleu 74.32, Thu Mar 21 03:05:32 2019
# Finished an epoch, step 4526. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 1186
    src: which is longer vojinović bridge or bharatanatyam
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Vojinović_Bridge || var_a = dbr_Bharatanatyam) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Vojinović_Bridge || var_a =
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:34 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:05:40 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4600 lr 1 step-time 0.03s wps 69.92K ppl 1.20 gN 1.11 bleu 74.32, Thu Mar 21 03:05:43 2019
# Finished an epoch, step 4612. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 1013
    src: is stefan starzyński monument in warsaw
    ref: ask where brack_open dbr_Stefan_Starzyński_Monument dbo_location dbr_Warsaw brack_close
    nmt: ask where brack_open dbr_La_Barre_Monument dbo_location dbr_Warsaw brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:45 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:05:50 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 4698. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 399
    src: give me the location of forest of the martyrs
    ref: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:05:55 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:06:00 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4700 lr 1 step-time 0.04s wps 62.66K ppl 1.20 gN 1.35 bleu 74.32, Thu Mar 21 03:06:02 2019
# Finished an epoch, step 4784. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 783
    src: what are the coordinates of shwezigon pagoda bell
    ref: select var_a where brack_open dbr_Shwezigon_Pagoda_Bell georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Shwezigon_Pagoda_Bell georss_point var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:05 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:06:10 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4800 lr 1 step-time 0.03s wps 68.40K ppl 1.17 gN 1.03 bleu 74.32, Thu Mar 21 03:06:12 2019
# Finished an epoch, step 4870. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 562
    src: what's rudi geodetic point native name
    ref: select var_a where brack_open dbr_Rudi_Geodetic_Point dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Rudi_Geodetic_Point dbp_nativeName var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:15 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:06:20 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 4900 lr 1 step-time 0.03s wps 72.03K ppl 1.17 gN 1.04 bleu 74.32, Thu Mar 21 03:06:22 2019
# Finished an epoch, step 4956. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
  # 697
    src: what is wild bill hickok memorial about
    ref: select var_a where brack_open dbr_Wild_Bill_Hickok_Memorial dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Wild_Bill_Hickok_Memorial dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-4000, time 0.02s
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:25 2019.
  bleu dev: 74.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 4000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:06:30 2019.
  bleu test: 79.6
  saving hparams to /tmp/nmt_model/hparams
  step 5000 lr 1 step-time 0.03s wps 67.90K ppl 1.16 gN 0.92 bleu 74.32, Thu Mar 21 03:06:33 2019
# Save eval, global step 5000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 548
    src: what's treaty of lausanne monument and museum native name
    ref: select var_a where brack_open dbr_Treaty_of_Lausanne_Monument_and_Museum dbp_nativeName var_a brack_close
    nmt: select var_a where brack_open dbr_Treaty_of_Lausanne_Monument_and_Museum dbp_nativeName var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  eval dev: perplexity 1.35, time 0s, Thu Mar 21 03:06:34 2019.
  eval test: perplexity 1.14, time 1s, Thu Mar 21 03:06:35 2019.
# Finished an epoch, step 5042. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 431
    src: where is monument to the battle of the nations located in
    ref: select var_a where brack_open dbr_Monument_to_the_Battle_of_the_Nations dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_the_Battle_of_the_Nations dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:37 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:06:43 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5100 lr 1 step-time 0.03s wps 72.29K ppl 1.15 gN 0.95 bleu 82.90, Thu Mar 21 03:06:46 2019
# Finished an epoch, step 5128. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 995
    src: is university of edinburgh older than carefree sundial
    ref: ask where brack_open dbr_University_of_Edinburgh dbp_complete var_a sep_dot dbr_Carefree_sundial dbp_complete var_b sep_dot FILTER(var_a math_lt var_b) brack_close
    nmt: ask where brack_open dbr_University_of_Portsmouth dbp_complete var_a sep_dot dbr_Carefree_sundial dbp_complete var_b sep_dot FILTER(var_a math_lt var_b) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:47 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:06:53 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5200 lr 1 step-time 0.03s wps 71.04K ppl 1.14 gN 0.92 bleu 82.90, Thu Mar 21 03:06:57 2019
# Finished an epoch, step 5214. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 1196
    src: which is longer pegasus and dragon or carpathian mountains
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Pegasus_and_Dragon || var_a = dbr_Carpathian_Mountains) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Pegasus_and_Dragon || var_a = dbr_Taq_Kasra) brack_close _oba_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:06:58 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:07:03 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5300 lr 1 step-time 0.03s wps 71.21K ppl 1.14 gN 1.09 bleu 82.90, Thu Mar 21 03:07:07 2019
# Finished an epoch, step 5300. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 1266
    src: give me the daryl palumbo tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Daryl_Palumbo
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:08 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:07:13 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 5386. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 1045
    src: is royal naval division memorial in london
    ref: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_London brack_close
    nmt: ask where brack_open dbr_Royal_Naval_Division_Memorial dbo_location dbr_Azerbaijan brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:18 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:07:23 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5400 lr 1 step-time 0.04s wps 62.07K ppl 1.13 gN 0.86 bleu 82.90, Thu Mar 21 03:07:25 2019
# Finished an epoch, step 5472. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 732
    src: what is llama de la libertad about
    ref: select var_a where brack_open dbr_Llama_de_la_Libertad dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Llama_de_la_Libertad dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:27 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:07:33 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5500 lr 1 step-time 0.03s wps 73.03K ppl 1.12 gN 0.81 bleu 82.90, Thu Mar 21 03:07:35 2019
# Finished an epoch, step 5558. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 188
    src: fort saint-elme
    ref: select var_a where brack_open dbr_Fort_Saint-Elme_(France) dbo_abstract var_a brack_close
    nmt: select var_a where brack_open
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:38 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:07:43 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5600 lr 1 step-time 0.03s wps 69.08K ppl 1.12 gN 0.71 bleu 82.90, Thu Mar 21 03:07:46 2019
# Finished an epoch, step 5644. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 666
    src: what is samadhi of ranjit singh all about
    ref: select var_a where brack_open dbr_Samadhi_of_Ranjit_Singh dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Samadhi_of_Ranjit_Singh dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:48 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:07:53 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5700 lr 1 step-time 0.03s wps 71.42K ppl 1.11 gN 0.82 bleu 82.90, Thu Mar 21 03:07:56 2019
# Finished an epoch, step 5730. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 86
    src: what is garghabazar caravanserai
    ref: select var_a where brack_open dbr_Garghabazar_Caravanserai dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Garghabazar_Caravanserai dbo_abstract var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:07:58 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:08:03 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5800 lr 1 step-time 0.03s wps 73.39K ppl 1.11 gN 0.78 bleu 82.90, Thu Mar 21 03:08:07 2019
# Finished an epoch, step 5816. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 510
    src: when was st. jago's arch completed
    ref: select var_a where brack_open dbr_St._Jago's_Arch dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_St._Jago's_Arch dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:08 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:08:13 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 5900 lr 1 step-time 0.03s wps 72.67K ppl 1.10 gN 0.76 bleu 82.90, Thu Mar 21 03:08:17 2019
# Finished an epoch, step 5902. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 482
    src: who designed goddess of democracy
    ref: select var_a where brack_open dbr_Goddess_of_Democracy_(Hong_Kong) dbo_designer var_a brack_close
    nmt: select var_a where brack_open dbr_Goddess_of_Democracy_(Hong_Kong) dbo_designer var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:18 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:08:23 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 5988. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
  # 247
    src: where is ramavarma appan thampuran memorial
    ref: select var_a where brack_open dbr_Ramavarma_Appan_Thampuran_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Ramavarma_Appan_Thampuran_Memorial dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-5000, time 0.02s
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:28 2019.
  bleu dev: 82.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 5000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:08:33 2019.
  bleu test: 90.3
  saving hparams to /tmp/nmt_model/hparams
  step 6000 lr 1 step-time 0.04s wps 62.57K ppl 1.10 gN 0.83 bleu 82.90, Thu Mar 21 03:08:35 2019
# Save eval, global step 6000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 531
    src: when was shrine of the book completed
    ref: select var_a where brack_open dbr_Shrine_of_the_Book dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Shrine_of_the_Book dbp_complete var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  eval dev: perplexity 1.31, time 0s, Thu Mar 21 03:08:35 2019.
  eval test: perplexity 1.10, time 1s, Thu Mar 21 03:08:37 2019.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 57
    src: is abraham lincoln a place
    ref: ask where brack_open dbr_Abraham_Lincoln_(Flannery) rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Abraham_Lincoln_(Flannery) rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:38 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:08:43 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 6074. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 154
    src: quailey's hill memorial
    ref: select var_a where brack_open dbr_Quailey's_Hill_Memorial dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Quailey's_Hill_Memorial dbo_abstract
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:47 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:08:52 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6100 lr 1 step-time 0.03s wps 73.15K ppl 1.10 gN 0.77 bleu 83.70, Thu Mar 21 03:08:54 2019
# Finished an epoch, step 6160. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 838
    src: how north is quailey's hill memorial
    ref: select var_a where brack_open dbr_Quailey's_Hill_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Quailey's_Hill_Memorial geo_lat var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:08:57 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:09:02 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6200 lr 1 step-time 0.03s wps 72.46K ppl 1.10 gN 0.75 bleu 83.70, Thu Mar 21 03:09:05 2019
# Finished an epoch, step 6246. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 180
    src: southern general cemetery
    ref: select var_a where brack_open dbr_Southern_General_Cemetery dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Southern_General_Cemetery dbo_abstract
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:07 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:09:12 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6300 lr 1 step-time 0.03s wps 73.96K ppl 1.09 gN 0.69 bleu 83.70, Thu Mar 21 03:09:15 2019
# Finished an epoch, step 6332. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 920
    src: longitude of 1820 settlers national monument
    ref: select var_a where brack_open dbr_1820_Settlers_National_Monument geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_1820_Settlers_National_Monument geo_long var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:16 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:09:21 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6400 lr 1 step-time 0.03s wps 72.05K ppl 1.09 gN 0.64 bleu 83.70, Thu Mar 21 03:09:25 2019
# Finished an epoch, step 6418. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 527
    src: when was blantyre monument completed
    ref: select var_a where brack_open dbr_Blantyre_Monument dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Blantyre_Monument dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:26 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:09:32 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6500 lr 1 step-time 0.03s wps 68.59K ppl 1.08 gN 0.61 bleu 83.70, Thu Mar 21 03:09:36 2019
# Finished an epoch, step 6504. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 862
    src: how north is farhad and shirin monument
    ref: select var_a where brack_open dbr_Farhad_and_Shirin_Monument geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Farhad_and_Shirin_Monument geo_lat var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:37 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:09:42 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 6590. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 75
    src: what is momine khatun mausoleum
    ref: select var_a where brack_open dbr_Momine_Khatun_Mausoleum dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Momine_Khatun_Mausoleum dbo_abstract var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:47 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:09:52 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6600 lr 1 step-time 0.04s wps 56.83K ppl 1.08 gN 0.60 bleu 83.70, Thu Mar 21 03:09:54 2019
# Finished an epoch, step 6676. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 656
    src: what is artillery memorial all about
    ref: select var_a where brack_open dbr_Artillery_Memorial,_Cape_Town dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Artillery_Memorial,_Cape_Town dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:09:57 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:10:03 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6700 lr 1 step-time 0.03s wps 69.23K ppl 1.08 gN 0.49 bleu 83.70, Thu Mar 21 03:10:06 2019
# Finished an epoch, step 6762. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 1081
    src: was dewey monument finished by <B>
    ref: ask where brack_open dbr_Dewey_Monument dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
    nmt: ask where brack_open dbr_Dewey_Monument dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.03s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:10:08 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:10:14 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6800 lr 1 step-time 0.03s wps 69.87K ppl 1.07 gN 0.49 bleu 83.70, Thu Mar 21 03:10:17 2019
# Finished an epoch, step 6848. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 399
    src: give me the location of forest of the martyrs
    ref: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:10:19 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:10:24 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 6900 lr 1 step-time 0.03s wps 71.17K ppl 1.07 gN 0.48 bleu 83.70, Thu Mar 21 03:10:28 2019
# Finished an epoch, step 6934. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
  # 608
    src: what is guanyin of the south china sea related to
    ref: select var_a where brack_open dbr_Guanyin_of_the_South_China_Sea,_Mount_Xiqiao dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Guanyin_of_the_South_China_Sea,_Mount_Xiqiao dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-6000, time 0.02s
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:10:29 2019.
  bleu dev: 83.7
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 6000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:10:35 2019.
  bleu test: 92.5
  saving hparams to /tmp/nmt_model/hparams
  step 7000 lr 1 step-time 0.04s wps 65.73K ppl 1.07 gN 0.43 bleu 83.70, Thu Mar 21 03:10:39 2019
# Save eval, global step 7000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 609
    src: what is yusif ibn kuseyir mausoleum related to
    ref: select var_a where brack_open dbr_Yusif_ibn_Kuseyir_Mausoleum dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Yusif_ibn_Kuseyir_Mausoleum dct_subject var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
  eval dev: perplexity 1.28, time 0s, Thu Mar 21 03:10:39 2019.
  eval test: perplexity 1.08, time 1s, Thu Mar 21 03:10:41 2019.
# Finished an epoch, step 7020. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 1304
    src: what do statue of lenin in kharkiv and statue of lenin in kharkiv have in common
    ref: select wildcard where brack_open brack_open dbr_Statue_of_Lenin_in_Kharkiv var_a var_b sep_dot dbr_Statue_of_Lenin_in_Kharkiv var_a var_b brack_close UNION brack_open brack_open dbr_Statue_of_Lenin_in_Kharkiv var_a var_b sep_dot dbr_Statue_of_Lenin_in_Kharkiv var_a var_b brack_close UNION brack_open var_c var_d dbr_Statue_of_Lenin_in_Kharkiv sep_dot var_c var_d dbr_Statue_of_Lenin_in_Kharkiv brack_close brack_close UNION brack_open var_c var_d dbr_Statue_of_Lenin_in_Kharkiv sep_dot var_c var_d dbr_Statue_of_Lenin_in_Kharkiv brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Statue_of_Lenin_in_Bila_Tserkva var_a var_b sep_dot dbr_Statue_of_Lenin_in_Bila_Tserkva var_a var_b brack_close UNION brack_open brack_open dbr_Statue_of_Lenin_in_Bila_Tserkva var_a var_b sep_dot dbr_Statue_of_Lenin_in_Bila_Tserkva var_a var_b brack_close UNION brack_open var_c var_d dbr_Statue_of_Lenin_in_Bila_Tserkva sep_dot var_c var_d
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:10:43 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:10:49 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7100 lr 1 step-time 0.04s wps 63.28K ppl 1.07 gN 0.41 bleu 84.85, Thu Mar 21 03:10:53 2019
# Finished an epoch, step 7106. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 297
    src: where can one find monument aux braves de n.d.g.
    ref: select var_a where brack_open dbr_Monument_aux_braves_de_N.D.G sep_dot dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_aux_braves_de_N.D.G sep_dot dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:10:54 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:00 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 7192. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 967
    src: when was haikou clock tower built
    ref: select var_a where brack_open dbr_Haikou_Clock_Tower dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Haikou_Clock_Tower dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:05 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:11 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7200 lr 1 step-time 0.04s wps 53.43K ppl 1.07 gN 0.49 bleu 84.85, Thu Mar 21 03:11:13 2019
# Finished an epoch, step 7278. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 1291
    src: what do obelisk of são paulo and obelisk of são paulo have in common
    ref: select wildcard where brack_open brack_open dbr_Obelisk_of_São_Paulo var_a var_b sep_dot dbr_Obelisk_of_São_Paulo var_a var_b brack_close UNION brack_open brack_open dbr_Obelisk_of_São_Paulo var_a var_b sep_dot dbr_Obelisk_of_São_Paulo var_a var_b brack_close UNION brack_open var_c var_d dbr_Obelisk_of_São_Paulo sep_dot var_c var_d dbr_Obelisk_of_São_Paulo brack_close brack_close UNION brack_open var_c var_d dbr_Obelisk_of_São_Paulo sep_dot var_c var_d dbr_Obelisk_of_São_Paulo brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame var_a var_b sep_dot dbr_Tomb_of_Anarkali var_a var_b brack_close UNION brack_open brack_open dbr_Tomb_of_Hayreddin_Barbarossa var_a var_b sep_dot dbr_Tomb_of_Anarkali var_a var_b brack_close UNION brack_open var_c var_d
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:16 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:22 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7300 lr 1 step-time 0.04s wps 65.57K ppl 1.07 gN 0.42 bleu 84.85, Thu Mar 21 03:11:24 2019
# Finished an epoch, step 7364. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 1085
    src: is university of liverpool more recent than carefree sundial
    ref: ask where brack_open dbr_University_of_Liverpool dbp_complete var_a sep_dot dbr_Carefree_sundial dbp_complete var_b sep_dot FILTER(var_a math_gt var_b) brack_close
    nmt: ask where brack_open dbr_University_of_Central_Lancashire dbp_complete var_a sep_dot dbr_Carefree_sundial dbp_complete var_b sep_dot FILTER(var_a math_gt var_b) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.03s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:27 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:32 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7400 lr 1 step-time 0.04s wps 68.21K ppl 1.06 gN 0.42 bleu 84.85, Thu Mar 21 03:11:35 2019
# Finished an epoch, step 7450. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 284
    src: where can one find fountain of neptune
    ref: select var_a where brack_open dbr_Fountain_of_Neptune,_Bologna dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Fountain_of_Neptune,_Bologna dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:37 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:43 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7500 lr 1 step-time 0.03s wps 69.32K ppl 1.06 gN 0.40 bleu 84.85, Thu Mar 21 03:11:46 2019
# Finished an epoch, step 7536. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 1093
    src: which is taller between ranevskaya monument and harvard bixi
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Harvard_Bixi) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Derlis_Paredes) brack_close _oba_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:48 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:11:54 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7600 lr 1 step-time 0.04s wps 66.29K ppl 1.06 gN 0.40 bleu 84.85, Thu Mar 21 03:11:58 2019
# Finished an epoch, step 7622. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 168
    src: cho huan lai memorial
    ref: select var_a where brack_open dbr_Cho_Huan_Lai_Memorial dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Cho_Huan_Lai_Memorial dbo_abstract var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:11:59 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:12:04 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7700 lr 1 step-time 0.04s wps 68.03K ppl 1.06 gN 0.44 bleu 84.85, Thu Mar 21 03:12:08 2019
# Finished an epoch, step 7708. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 603
    src: what is monument to nizami ganjavi in tashkent related to
    ref: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Tashkent dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Tashkent dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:12:09 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:12:15 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 7794. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 895
    src: longitude of obelisk of são paulo
    ref: select var_a where brack_open dbr_Obelisk_of_São_Paulo geo_long var_a brack_close
    nmt: select var_a where brack_open dbr_Obelisk_of_São_Paulo geo_long var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:12:21 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:12:26 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7800 lr 1 step-time 0.04s wps 55.87K ppl 1.06 gN 0.45 bleu 84.85, Thu Mar 21 03:12:28 2019
# Finished an epoch, step 7880. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 332
    src: location of convento de los agustinos
    ref: select var_a where brack_open dbr_Convento_de_los_Agustinos dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Convento_de_los_Agustinos dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:12:32 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:12:38 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 7900 lr 1 step-time 0.04s wps 65.14K ppl 1.06 gN 0.41 bleu 84.85, Thu Mar 21 03:12:40 2019
# Finished an epoch, step 7966. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
  # 242
    src: where is lord murugan statue
    ref: select var_a where brack_open dbr_Lord_Murugan_Statue dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Lord_Murugan_Statue dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-7000, time 0.02s
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:12:43 2019.
  bleu dev: 84.9
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 7000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:12:48 2019.
  bleu test: 94.1
  saving hparams to /tmp/nmt_model/hparams
  step 8000 lr 1 step-time 0.04s wps 68.64K ppl 1.05 gN 0.40 bleu 84.85, Thu Mar 21 03:12:51 2019
# Save eval, global step 8000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 596
    src: what is holy trinity column in olomouc related to
    ref: select var_a where brack_open dbr_Holy_Trinity_Column_in_Olomouc dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Holy_Trinity_Column_in_Olomouc dct_subject var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:12:52 2019.
  eval test: perplexity 1.07, time 1s, Thu Mar 21 03:12:53 2019.
# Finished an epoch, step 8052. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 1222
    src: which is longer los angeles police department memorial for fallen officers or luigi albani
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Los_Angeles_Police_Department_Memorial_for_Fallen_Officers || var_a = dbr_Luigi_Albani) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Los_Angeles_Police_Department_Memorial_for_Fallen_Officers || var_a = dbr_Aaron_Smith_(magician)) brack_close _oba_ var_b limit 1
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:12:56 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:13:03 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8100 lr 1 step-time 0.04s wps 60.13K ppl 1.06 gN 0.40 bleu 85.53, Thu Mar 21 03:13:06 2019
# Finished an epoch, step 8138. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 747
    src: what are the coordinates of garibaldi monument in taganrog
    ref: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Garibaldi_Monument_in_Taganrog georss_point var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.04s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:13:09 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:13:15 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8200 lr 1 step-time 0.04s wps 60.64K ppl 1.05 gN 0.43 bleu 85.53, Thu Mar 21 03:13:18 2019
# Finished an epoch, step 8224. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 399
    src: give me the location of forest of the martyrs
    ref: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Forest_of_the_Martyrs dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:13:20 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:13:26 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8300 lr 1 step-time 0.04s wps 67.40K ppl 1.05 gN 0.41 bleu 85.53, Thu Mar 21 03:13:30 2019
# Finished an epoch, step 8310. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 282
    src: where can one find victims of iași pogrom monument
    ref: select var_a where brack_open dbr_Victims_of_Iași_Pogrom_Monument dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Victims_of_Iași_Pogrom_Monument dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:13:31 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:13:37 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 8396. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 1175
    src: which is longer wat's dyke or sagadahoc bridge
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Wat's_Dyke || var_a = dbr_Sagadahoc_Bridge) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Wat's_Dyke || var_a = dbr_Aaron_Smith_(magician)) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:13:43 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:13:49 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8400 lr 1 step-time 0.04s wps 54.26K ppl 1.06 gN 1.31 bleu 85.53, Thu Mar 21 03:13:51 2019
# Finished an epoch, step 8482. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 476
    src: where is monument to zagir ismagilov located in
    ref: select var_a where brack_open dbr_Monument_to_Zagir_Ismagilov dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Zagir_Ismagilov dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:13:55 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:14:01 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8500 lr 1 step-time 0.04s wps 55.69K ppl 1.08 gN 0.87 bleu 85.53, Thu Mar 21 03:14:03 2019
# Finished an epoch, step 8568. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 391
    src: give me the location of gibbet of montfaucon
    ref: select var_a where brack_open dbr_Gibbet_of_Montfaucon dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Gibbet_of_Montfaucon dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:14:06 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:14:12 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8600 lr 1 step-time 0.04s wps 62.87K ppl 1.05 gN 0.50 bleu 85.53, Thu Mar 21 03:14:15 2019
# Finished an epoch, step 8654. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 165
    src: civil rights memorial
    ref: select var_a where brack_open dbr_Civil_Rights_Memorial dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Civil_Rights_Memorial dbo_abstract
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:14:18 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:14:24 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8700 lr 1 step-time 0.04s wps 64.78K ppl 1.05 gN 0.43 bleu 85.53, Thu Mar 21 03:14:27 2019
# Finished an epoch, step 8740. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 525
    src: when was saint-vincent gate completed
    ref: select var_a where brack_open dbr_Saint-Vincent_Gate dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Saint-Vincent_Gate dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:14:29 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:14:34 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8800 lr 1 step-time 0.04s wps 68.00K ppl 1.05 gN 0.41 bleu 85.53, Thu Mar 21 03:14:38 2019
# Finished an epoch, step 8826. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 324
    src: location of grotto of our lady of lourdes
    ref: select var_a where brack_open dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:14:39 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:14:45 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 8900 lr 1 step-time 0.03s wps 68.70K ppl 1.05 gN 0.39 bleu 85.53, Thu Mar 21 03:14:49 2019
# Finished an epoch, step 8912. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
  # 1054
    src: was cross of all nations finished by <B>
    ref: ask where brack_open dbr_Cross_of_All_Nations dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
    nmt: ask where brack_open dbr_Cross_of_All_Nations dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.02s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:14:50 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:14:55 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 8998. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
  # 69
    src: is mausoleum of sheikh juneyd a place
    ref: ask where brack_open dbr_Mausoleum_of_Sheikh_Juneyd rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Mausoleum_of_Sheikh_Juneyd rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-8000, time 0.03s
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 1s, Thu Mar 21 03:15:02 2019.
  bleu dev: 85.5
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 8000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Thu Mar 21 03:15:09 2019.
  bleu test: 94.7
  saving hparams to /tmp/nmt_model/hparams
  step 9000 lr 1 step-time 0.05s wps 47.21K ppl 1.05 gN 0.44 bleu 85.53, Thu Mar 21 03:15:11 2019
# Save eval, global step 9000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.04s
  # 1241
    src: give me the fourth of july tomato tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Fourth_of_July_tomato
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Flag_of_Estonia
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.04s
  eval dev: perplexity 1.28, time 0s, Thu Mar 21 03:15:12 2019.
  eval test: perplexity 1.06, time 2s, Thu Mar 21 03:15:14 2019.
# Finished an epoch, step 9084. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 614
    src: what is shaka memorial related to
    ref: select var_a where brack_open dbr_Shaka_Memorial dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Shaka_Memorial dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:15:18 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:15:23 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9100 lr 1 step-time 0.04s wps 59.44K ppl 1.05 gN 0.39 bleu 85.53, Thu Mar 21 03:15:26 2019
# Finished an epoch, step 9170. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 63
    src: is forest of the martyrs a protected area
    ref: ask where brack_open dbr_Forest_of_the_Martyrs rdf_type dbo_ProtectedArea brack_close
    nmt: ask where brack_open dbr_Birth_of_the_New_World rdf_type dbo_Artwork brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:15:29 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:15:34 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9200 lr 1 step-time 0.04s wps 65.83K ppl 1.04 gN 0.38 bleu 85.53, Thu Mar 21 03:15:37 2019
# Finished an epoch, step 9256. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 1009
    src: is blantyre monument in renfrewshire
    ref: ask where brack_open dbr_Blantyre_Monument dbo_location dbr_Renfrewshire brack_close
    nmt: ask where brack_open dbr_Blantyre_Monument dbo_location dbr_Turkey brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:15:40 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:15:45 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9300 lr 1 step-time 0.04s wps 60.92K ppl 1.04 gN 0.36 bleu 85.53, Thu Mar 21 03:15:49 2019
# Finished an epoch, step 9342. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 1181
    src: which is longer ahu akivi or jon foo
    ref: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Ahu_Akivi || var_a = dbr_Jon_Foo) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_length var_b sep_dot FILTER(var_a = dbr_Vojinović_Bridge || var_a = dbr_Holy_Trinity_Church_(Nashville)) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:15:51 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:15:57 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9400 lr 1 step-time 0.04s wps 61.02K ppl 1.04 gN 0.37 bleu 85.53, Thu Mar 21 03:16:01 2019
# Finished an epoch, step 9428. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 515
    src: when was conolly's folly completed
    ref: select var_a where brack_open dbr_Conolly's_Folly dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Conolly's_Folly dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:03 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:16:09 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9500 lr 1 step-time 0.04s wps 59.72K ppl 1.04 gN 0.40 bleu 85.53, Thu Mar 21 03:16:13 2019
# Finished an epoch, step 9514. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 217
    src: where is wild bill hickok memorial
    ref: select var_a where brack_open dbr_Wild_Bill_Hickok_Memorial dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Wild_Bill_Hickok_Memorial dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:14 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:16:20 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9600 lr 1 step-time 0.04s wps 64.66K ppl 1.04 gN 0.40 bleu 85.53, Thu Mar 21 03:16:24 2019
# Finished an epoch, step 9600. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 683
    src: what is khanegah tomb about
    ref: select var_a where brack_open dbr_Khanegah_tomb dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Khanegah_tomb dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:25 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:16:31 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 9686. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 1239
    src: give me the yuhi falls tallest <B>
    ref: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b limit dbr_Yuhi_Falls
    nmt: select var_a where brack_open var_a rdf_type <B> sep_dot var_a dbp_height var_b brack_close _obd_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:36 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:16:42 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9700 lr 1 step-time 0.04s wps 52.44K ppl 1.04 gN 0.36 bleu 85.53, Thu Mar 21 03:16:44 2019
# Finished an epoch, step 9772. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
  # 225
    src: where is st. jago's arch
    ref: select var_a where brack_open dbr_St._Jago's_Arch dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_St._Jago's_Arch dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.03s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:48 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:16:54 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9800 lr 1 step-time 0.04s wps 58.78K ppl 1.04 gN 0.35 bleu 85.53, Thu Mar 21 03:16:57 2019
# Finished an epoch, step 9858. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 40
    src: is dr. william d. young memorial a place
    ref: ask where brack_open dbr_Dr._William_D._Young_Memorial rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Dr._William_D._Young_Memorial rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:16:59 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:17:05 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 9900 lr 1 step-time 0.04s wps 64.99K ppl 1.04 gN 0.37 bleu 85.53, Thu Mar 21 03:17:08 2019
# Finished an epoch, step 9944. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
  # 638
    src: what is monument to nizami ganjavi in baku all about
    ref: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Baku dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Nizami_Ganjavi_in_Baku dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-9000, time 0.02s
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:17:10 2019.
  bleu dev: 85.3
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 9000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:17:16 2019.
  bleu test: 95.2
  saving hparams to /tmp/nmt_model/hparams
  step 10000 lr 1 step-time 0.04s wps 68.16K ppl 1.04 gN 0.40 bleu 85.53, Thu Mar 21 03:17:20 2019
# Save eval, global step 10000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 77
    src: what is chartered company monument
    ref: select var_a where brack_open dbr_Chartered_Company_Monument dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Chartered_Company_Monument dbo_abstract var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:17:20 2019.
  eval test: perplexity 1.05, time 1s, Thu Mar 21 03:17:22 2019.
# Finished an epoch, step 10030. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 789
    src: latitude of dewey arch
    ref: select var_a where brack_open dbr_Dewey_Arch geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Dewey_Arch geo_lat var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:17:24 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:17:30 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10100 lr 1 step-time 0.04s wps 60.67K ppl 1.04 gN 0.36 bleu 85.58, Thu Mar 21 03:17:34 2019
# Finished an epoch, step 10116. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.04s
  # 485
    src: who designed wilfrid laurier memorial
    ref: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial dbo_designer var_a brack_close
    nmt: select var_a where brack_open dbr_Wilfrid_Laurier_Memorial dbo_designer var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:17:36 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:17:42 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10200 lr 1 step-time 0.04s wps 55.40K ppl 1.04 gN 0.42 bleu 85.58, Thu Mar 21 03:17:47 2019
# Finished an epoch, step 10202. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 368
    src: location of indio comahue monument
    ref: select var_a where brack_open dbr_Indio_Comahue_Monument dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Indio_Comahue_Monument dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:17:48 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:17:54 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 10288. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 284
    src: where can one find fountain of neptune
    ref: select var_a where brack_open dbr_Fountain_of_Neptune,_Bologna dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Fountain_of_Neptune,_Bologna dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:18:00 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:18:07 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10300 lr 1 step-time 0.05s wps 48.90K ppl 1.03 gN 0.39 bleu 85.58, Thu Mar 21 03:18:09 2019
# Finished an epoch, step 10374. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 494
    src: who designed old city hall cenotaph
    ref: select var_a where brack_open dbr_Old_City_Hall_Cenotaph,_Toronto dbo_designer var_a brack_close
    nmt: select var_a where brack_open dbr_Old_City_Hall_Cenotaph,_Toronto dbo_designer var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:18:13 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:18:19 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10400 lr 1 step-time 0.04s wps 60.97K ppl 1.03 gN 0.34 bleu 85.58, Thu Mar 21 03:18:22 2019
# Finished an epoch, step 10460. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 47
    src: is garghabazar mosque a religious building
    ref: ask where brack_open dbr_Garghabazar_Mosque rdf_type dbo_ReligiousBuilding brack_close
    nmt: ask where brack_open dbr_Garghabazar_Mosque rdf_type dbo_Building brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:18:25 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:18:32 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10500 lr 1 step-time 0.04s wps 61.91K ppl 1.03 gN 0.37 bleu 85.58, Thu Mar 21 03:18:35 2019
# Finished an epoch, step 10546. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.02s
  # 1139
    src: which is taller between newkirk viaduct monument and international relations of the great powers
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_International_relations_of_the_Great_Powers_(1814–1919)) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Monumento_a_la_abolición_de_la_esclavitud) brack_close _oba_ var_b limit 1
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:18:37 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:18:44 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10600 lr 1 step-time 0.04s wps 60.10K ppl 1.03 gN 0.37 bleu 85.58, Thu Mar 21 03:18:48 2019
# Finished an epoch, step 10632. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 1136
    src: which is taller between newkirk viaduct monument and united states congressional delegations from kentucky
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_United_States_congressional_delegations_from_Kentucky) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Newkirk_Viaduct_Monument || var_a = dbr_Felix_Sturm_vs._Oscar_De_La_Hoya) brack_close _oba_ var_b limit 1
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:18:50 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Thu Mar 21 03:18:58 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10700 lr 1 step-time 0.04s wps 54.40K ppl 1.03 gN 0.35 bleu 85.58, Thu Mar 21 03:19:02 2019
# Finished an epoch, step 10718. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 10
    src: is chartreuse du liget a religious building
    ref: ask where brack_open dbr_Chartreuse_du_Liget rdf_type dbo_ReligiousBuilding brack_close
    nmt: ask where brack_open dbr_Chartreuse_du_Liget rdf_type dbo_Person brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:19:04 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:19:11 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10800 lr 1 step-time 0.04s wps 57.00K ppl 1.03 gN 0.35 bleu 85.58, Thu Mar 21 03:19:16 2019
# Finished an epoch, step 10804. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 1075
    src: was old city hall cenotaph finished by <B>
    ref: ask where brack_open dbr_Old_City_Hall_Cenotaph,_Toronto dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
    nmt: ask where brack_open dbr_Old_City_Hall_Cenotaph,_Toronto dbp_complete var_a sep_dot FILTER(var_a math_leq <B>) brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:19:17 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:19:24 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 10890. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 1093
    src: which is taller between ranevskaya monument and harvard bixi
    ref: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Harvard_Bixi) brack_close _oba_ var_b limit 1
    nmt: select var_a where brack_open var_a dbp_height var_b sep_dot FILTER(var_a = dbr_Ranevskaya_Monument || var_a = dbr_Graeham_Goble) brack_close _oba_ var_b
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:19:30 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 7s, Thu Mar 21 03:19:37 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 10900 lr 1 step-time 0.05s wps 50.79K ppl 1.03 gN 0.40 bleu 85.58, Thu Mar 21 03:19:39 2019
# Finished an epoch, step 10976. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
  # 442
    src: where is memorial to victims of stalinist repression located in
    ref: select var_a where brack_open dbr_Memorial_to_Victims_of_Stalinist_Repression dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Memorial_to_Victims_of_Stalinist_Repression dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-10000, time 0.03s
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:19:43 2019.
  bleu dev: 85.6
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 10000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:19:50 2019.
  bleu test: 95.9
  saving hparams to /tmp/nmt_model/hparams
  step 11000 lr 1 step-time 0.05s wps 53.21K ppl 1.03 gN 0.34 bleu 85.58, Thu Mar 21 03:19:53 2019
# Save eval, global step 11000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 203
    src: monument to endre ady
    ref: select var_a where brack_open dbr_Monument_to_Endre_Ady,_Zalău dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Monument_to_Endre_Ady,_Zalău dbo_abstract var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:19:54 2019.
  eval test: perplexity 1.04, time 2s, Thu Mar 21 03:19:56 2019.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 337
    src: location of imamzadeh
    ref: select var_a where brack_open dbr_Imamzadeh_(Ganja) dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Imamzadeh_(Ganja) dbo_location
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:19:57 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:20:05 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 11062. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 87
    src: what is basilica in qum village
    ref: select var_a where brack_open dbr_Basilica_in_Qum_village dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Basilica_in_Qum_village dbo_abstract var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:20:09 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 6s, Thu Mar 21 03:20:15 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11100 lr 1 step-time 0.04s wps 59.05K ppl 1.03 gN 0.36 bleu 86.25, Thu Mar 21 03:20:18 2019
# Finished an epoch, step 11148. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 195
    src: khojaly massacre memorials
    ref: select var_a where brack_open dbr_Khojaly_massacre_memorials dbo_abstract var_a brack_close
    nmt: select var_a where brack_open dbr_Khojaly_massacre_memorials dbo_abstract
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:20:21 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:20:26 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11200 lr 1 step-time 0.04s wps 66.82K ppl 1.03 gN 0.33 bleu 86.25, Thu Mar 21 03:20:30 2019
# Finished an epoch, step 11234. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 825
    src: latitude of rhodes memorial
    ref: select var_a where brack_open dbr_Rhodes_Memorial geo_lat var_a brack_close
    nmt: select var_a where brack_open dbr_Rhodes_Memorial geo_lat var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:20:32 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:20:37 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11300 lr 1 step-time 0.03s wps 68.58K ppl 1.03 gN 0.34 bleu 86.25, Thu Mar 21 03:20:41 2019
# Finished an epoch, step 11320. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 606
    src: what is general philip sheridan related to
    ref: select var_a where brack_open dbr_General_Philip_Sheridan dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_General_Philip_Sheridan dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:20:42 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:20:48 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11400 lr 1 step-time 0.04s wps 68.44K ppl 1.03 gN 0.36 bleu 86.25, Thu Mar 21 03:20:52 2019
# Finished an epoch, step 11406. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 412
    src: give me the location of bhasha smritistambha
    ref: select var_a where brack_open dbr_Bhasha_Smritistambha dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Bhasha_Smritistambha dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:20:53 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:20:58 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
# Finished an epoch, step 11492. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 14
    src: is monument to ferdinand i a place
    ref: ask where brack_open dbr_Monument_to_Ferdinand_I rdf_type dbo_Place brack_close
    nmt: ask where brack_open dbr_Monument_to_Ferdinand_I rdf_type dbo_Place brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:03 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:21:09 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11500 lr 1 step-time 0.04s wps 57.71K ppl 1.03 gN 0.35 bleu 86.25, Thu Mar 21 03:21:11 2019
# Finished an epoch, step 11578. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 693
    src: what is kemal atatürk memorial about
    ref: select var_a where brack_open dbr_Kemal_Atatürk_Memorial,_Canberra dct_subject var_a brack_close
    nmt: select var_a where brack_open dbr_Kemal_Atatürk_Memorial,_Canberra dct_subject var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:14 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:21:20 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11600 lr 1 step-time 0.04s wps 65.26K ppl 1.03 gN 0.32 bleu 86.25, Thu Mar 21 03:21:22 2019
# Finished an epoch, step 11664. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 959
    src: building date of emperors yan and huang
    ref: select var_a where brack_open dbr_Emperors_Yan_and_Huang dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Emperors_Yan_and_Huang dbp_complete var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:25 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:21:30 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11700 lr 1 step-time 0.03s wps 71.30K ppl 1.02 gN 0.31 bleu 86.25, Thu Mar 21 03:21:33 2019
# Finished an epoch, step 11750. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
  # 302
    src: where can one find confederate memorial
    ref: select var_a where brack_open dbr_Confederate_Memorial_(Wilmington,_North_Carolina) dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Confederate_Memorial_(Wilmington,_North_Carolina) dbo_location var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.02s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:35 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 4s, Thu Mar 21 03:21:40 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11800 lr 1 step-time 0.03s wps 69.59K ppl 1.02 gN 0.35 bleu 86.25, Thu Mar 21 03:21:43 2019
# Finished an epoch, step 11836. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 1291
    src: what do obelisk of são paulo and obelisk of são paulo have in common
    ref: select wildcard where brack_open brack_open dbr_Obelisk_of_São_Paulo var_a var_b sep_dot dbr_Obelisk_of_São_Paulo var_a var_b brack_close UNION brack_open brack_open dbr_Obelisk_of_São_Paulo var_a var_b sep_dot dbr_Obelisk_of_São_Paulo var_a var_b brack_close UNION brack_open var_c var_d dbr_Obelisk_of_São_Paulo sep_dot var_c var_d dbr_Obelisk_of_São_Paulo brack_close brack_close UNION brack_open var_c var_d dbr_Obelisk_of_São_Paulo sep_dot var_c var_d dbr_Obelisk_of_São_Paulo brack_close brack_close
    nmt: select wildcard where brack_open brack_open dbr_Obelisk_of_São_Paulo var_a var_b sep_dot dbr_Obelisk_of_São_Paulo var_a var_b brack_close UNION brack_open brack_open dbr_Tomb_of_Hayreddin_Barbarossa var_a var_b sep_dot dbr_Grotto_of_Our_Lady_of_Lourdes,_Notre_Dame var_a var_b brack_close UNION brack_open var_c var_d
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:45 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:21:51 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 11900 lr 1 step-time 0.03s wps 68.36K ppl 1.03 gN 0.35 bleu 86.25, Thu Mar 21 03:21:55 2019
# Finished an epoch, step 11922. Perform external evaluation
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
  # 748
    src: what are the coordinates of khojaly genocide memorial
    ref: select var_a where brack_open dbr_Khojaly_Genocide_Memorial_(Baku) georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Khojaly_Genocide_Memorial_(Baku) georss_point var_a brack_close
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-11000, time 0.03s
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:21:56 2019.
  bleu dev: 86.2
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 11000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:22:02 2019.
  bleu test: 96.2
  saving hparams to /tmp/nmt_model/hparams
  step 12000 lr 1 step-time 0.04s wps 66.79K ppl 1.02 gN 0.35 bleu 86.25, Thu Mar 21 03:22:06 2019
# Save eval, global step 12000
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  # 736
    src: what are the coordinates of dewey monument
    ref: select var_a where brack_open dbr_Dewey_Monument georss_point var_a brack_close
    nmt: select var_a where brack_open dbr_Dewey_Monument georss_point var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:22:07 2019.
  eval test: perplexity 1.04, time 1s, Thu Mar 21 03:22:08 2019.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  # 965
    src: when was tomb of payava built
    ref: select var_a where brack_open dbr_Tomb_of_Payava dbp_complete var_a brack_close
    nmt: select var_a where brack_open dbr_Tomb_of_Payava dbp_complete var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.02s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:22:09 2019.
  eval test: perplexity 1.04, time 1s, Thu Mar 21 03:22:11 2019.
  loaded infer model parameters from /tmp/nmt_model/translate.ckpt-12000, time 0.03s
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:22:11 2019.
  bleu dev: 86.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:22:17 2019.
  bleu test: 96.4
  saving hparams to /tmp/nmt_model/hparams
# Final, step 12000 lr 1 step-time 0.04s wps 66.79K ppl 1.02 gN 0.35 dev ppl 1.27, dev bleu 86.4, test ppl 1.04, test bleu 96.4, Thu Mar 21 03:22:18 2019
# Done training!, time 1444s, Thu Mar 21 03:22:18 2019.
# Start evaluating saved best models.
  loaded infer model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.02s
  # 376
    src: give me the location of millennium monument of brest
    ref: select var_a where brack_open dbr_Millennium_Monument_of_Brest dbo_location var_a brack_close
    nmt: select var_a where brack_open dbr_Millennium_Monument_of_Brest dbo_location var_a brack_close
  loaded eval model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.02s
  eval dev: perplexity 1.27, time 0s, Thu Mar 21 03:22:18 2019.
  eval test: perplexity 1.04, time 1s, Thu Mar 21 03:22:20 2019.
  loaded infer model parameters from /tmp/nmt_model/best_bleu/translate.ckpt-12000, time 0.02s
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_dev
  done, num sentences 1328, num translations per input 1, time 0s, Thu Mar 21 03:22:20 2019.
  bleu dev: 86.4
  saving hparams to /tmp/nmt_model/hparams
# External evaluation, global step 12000
  decoding to output /tmp/nmt_model/output_test
  done, num sentences 10625, num translations per input 1, time 5s, Thu Mar 21 03:22:26 2019.
  bleu test: 96.4
  saving hparams to /tmp/nmt_model/hparams
# Best bleu, step 12000 lr 1 step-time 0.04s wps 66.79K ppl 1.02 gN 0.35 dev ppl 1.27, dev bleu 86.4, test ppl 1.04, test bleu 96.4, Thu Mar 21 03:22:27 2019
