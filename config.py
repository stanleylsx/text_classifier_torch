# -*- coding: utf-8 -*-
# @Time : 2021/07/15 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm

# 模式
# Mode
# train_classifier: 训练分类器
# train_classifier: Train a classifier
# interactive_predict: 交互模式
# interactive_predict: Interactive prediction mode
# test: 跑测试集
# test: Run on the test set
# convert_torch_script: 将torch模型保存torch script文件
# convert_torch_script: Save a Torch model as a Torch Script file
# show_model_info: 打印模型参数
# show_model_info: Display model parameters
mode = 'train_classifier'

# 阶段
# Stage
# finetune: 微调预训练模型
# finetune: Fine-tune the pre-trained model
# train_small_model: 单独训练小模型
# train_small_model: Train a small model independently
# distillation: 蒸馏
# distillation: Knowledge distillation
# prune: 模型剪枝
# prune: Model pruning
stage = 'finetune'

# 支持的模型
# Supported models
support_model = {
    'pretrained_types': ['Bert', 'DistilBert', 'RoBerta', 'ALBert', 'XLNet', 'Electra', 'MiniLM', 'DeBertaV3', 'XLM-RoBERTa'],
    # 预训练模型类型
    # Types of pre-trained models
    'ordinary_types': ['FastText', 'TextCNN', 'TextRNN', 'TextRCNN', 'Transformer']
    # 常规模型类型
    # Types of ordinary models
}

# 使用GPU设备
# Use GPU device
use_cuda = True
cuda_device = -1

train_configure = {
    # 训练数据集
    # Training dataset
    'train_file': 'datasets/example_datasets1/train_dataset.csv',
    # 验证数据集
    # Validation dataset
    'val_file': 'datasets/example_datasets1/val_dataset.csv',
    # 是否多标签分类
    # Whether multi-label classification
    'multilabel': False,
    # 使用交叉验证
    # Use cross-validation
    'kfold': False,
    'fold_splits': 4,
    # 没有验证集时，从训练集抽取验证集比例
    # Proportion of training data used as validation if no validation set is provided
    'validation_rate': 0.15,
    # 测试数据集
    # Test dataset
    'test_file': '',
    # token粒度,token选择字粒度的时候，词嵌入无效
    # Token granularity; word embedding is ineffective when token level is character-based
    # 词粒度:'word'
    # Word level: 'word'
    # 字粒度:'char'
    # Character level: 'char'
    'token_level': 'char',
    # 去停用词，仅限非预训练微调使用，字粒度不建议用
    # Remove stopwords; applies only to non-pretrained fine-tuning, not recommended for character-based token level
    'stop_words': False,
    'stop_words_file': 'datasets/example_datasets1/stop_words.txt',
    # 是否去掉特殊字符
    # Whether to remove special characters
    'remove_special': True,
    # 存放词表的地方，使用预训练模型的时候留空
    # Location of vocabulary file; leave empty when using a pre-trained model
    'token_file': 'datasets/example_datasets1/token2id_char',
    # 类别列表
    # List of classes
    'classes': ['negative', 'positive'],
    # 模型保存的文件夹
    # Folder for saving model checkpoints
    'checkpoints_dir': 'checkpoints/example1',
    # 句子最大长度
    # Maximum sentence length
    'max_sequence_length': 50,
    # 遗忘率
    # Dropout rate
    'dropout_rate': 0.5,
    # 随机种子
    # Random seed
    'seed': 3407,
    # 预训练模型是否前置加入Noisy
    # Whether to apply Noisy tuning to the pre-trained model
    'noisy_tune': False,
    # 若为二分类则使用binary
    # Use 'binary' for binary classification
    # 多分类使用micro或macro
    # Use 'micro' or 'macro' for multi-class classification
    'metrics_average': 'binary',
    # 类别样本比例失衡的时候可以考虑使用
    # Consider using this for imbalanced class samples
    'use_focal_loss': True,
    # focal loss的各个标签权重
    # Weights for each label in Focal Loss
    'weight': None,
    # 使用Poly Loss
    # Use Poly Loss
    'use_poly_loss': False,
    # 使用标签平滑
    # Use label smoothing
    'use_label_smoothing': False,
    'smooth_factor': 0.1,
    # 使用对抗学习
    # Use adversarial training
    'use_gan': True,
    # fgm: Fast Gradient Method
    # pgd: Projected Gradient Descent
    'gan_method': 'pgd',
    # 对抗次数
    # Number of adversarial rounds
    'attack_round': 3,
    # 使用对比学习，不推荐和对抗方法一起使用，效率慢收益不大
    # Use contrastive learning; not recommended with adversarial methods as it's less efficient with minor benefits
    'use_r_drop': False,
    # 使用Multisample Dropout
    # Use Multisample Dropout
    # 使用Multisample Dropout后dropout会失效
    # Dropout will be disabled when Multisample Dropout is used
    'multisample_dropout': True,
    'dropout_round': 5,
    # 是否进行warmup
    # Whether to use warmup
    'warmup': True,
    # warmup方法，可选：linear、cosine
    # Warmup method; options: 'linear', 'cosine'
    'scheduler_type': 'linear',
    # warmup步数，-1自动推断为总步数的0.1
    # Warmup steps; -1 infers as 10% of total steps
    'num_warmup_steps': -1,
    # 是否进行随机权重平均swa
    # Whether to use stochastic weight averaging (SWA)
    'swa': False,
    'swa_start_step': 5000,
    'swa_lr': 1e-6,
    # 每个多久平均一次
    # Frequency of averaging
    'anneal_epochs': 1,
    # 使用EMA
    # Use EMA (Exponential Moving Average)
    'ema': False,
    # 优化器选择
    # Optimizer selection
    'optimizer': 'AdamW',
    # 执行权重初始化，仅限于非微调
    # Perform weight initialization; only applies to non-fine-tuning
    'init_network': False,
    # 权重初始化方式，可选：xavier、kaiming、normal
    # Weight initialization method; options: 'xavier', 'kaiming', 'normal'
    'init_network_method': 'xavier',
    # fp16混合精度训练，仅在Cuda支持下使用
    # Use fp16 mixed-precision training; only applicable if CUDA is supported
    'use_fp16': False,

    # 预训练模型类型
    # Pre-trained model type
    'f_model_type': 'Bert',
    # 预训练模型细分类
    # Subtype of pre-trained model
    'ptm': 'bert-base-chinese',
    # 微调阶段的epoch
    # Epochs for the fine-tuning stage
    'f_epoch': 30,
    # 微调阶段的batch_size
    # Batch size for the fine-tuning stage
    'f_batch_size': 16,
    # 微调阶段的学习率
    # Learning rate for the fine-tuning stage
    'f_learning_rate': 4e-5,
    # 微调阶段每print_per_batch打印
    # Print frequency during the fine-tuning stage
    'f_print_per_batch': 50,
    # 是否提前结束微调
    # Whether to stop fine-tuning early
    'f_is_early_stop': True,
    # 微调阶段的patient
    # Patience for the fine-tuning stage
    'f_patient': 3,
    # Bert模型的名字
    # Name of the Bert model
    'f_model_name': 'finetune_bert_pgd.pkl',

    # 小模型的类型
    # Type of small model
    's_model_type': 'TextCNN',
    # Embedding向量维度
    # Dimension of embedding vectors
    'embedding_dim': 300,
    # TextCNN卷积核的个数
    # Number of filters in TextCNN
    'num_filters': 64,
    # TextRNN单个RNN的隐藏神经元个数
    # Number of hidden neurons in a single RNN of TextRNN
    # Transformer中Feed-Forward层的隐藏层维度
    # Hidden layer dimension in the Feed-Forward layer of Transformer
    'hidden_dim': 2048,
    # 使用attention
    # Use attention
    'use_attention': False,
    # 编码器个数(使用Transformer需要设定)
    # Number of encoders (required for Transformer)
    'encoder_num': 1,
    # 多头注意力的个数(使用Transformer需要设定)
    # Number of heads in multi-head attention (required for Transformer)
    'head_num': 12,
    # 单独小模型训练阶段的epoch
    # Epochs for independent training of the small model
    's_epoch': 100,
    # 单独小模型训练阶段的batch_size
    # Batch size for independent training of the small model
    's_batch_size': 16,
    # 单独小模型训练阶段的学习率
    # Learning rate for independent training of the small model
    's_learning_rate': 0.001,
    # 单独小模型训练阶段每print_per_batch打印
    # Print frequency during independent training of the small model
    's_print_per_batch': 50,
    # 单独小模型训练是否提前结束微调
    # Whether to stop training the small model early
    's_is_early_stop': True,
    # 单独小模型训练阶段的patient
    # Patience for the small model's training
    's_patient': 8,
    # 小模型的名字
    # Name of the small model
    's_model_name': 'torch.bin',
}

distill_configure = {
    # 自蒸馏
    # Self-distillation
    'self_distillation': True,
    # 蒸馏方法
    # Distillation method
    # ce: 最原始的蒸馏办法
    # ce: The original distillation method
    # kl: 论文：Distilling the Knowledge in a Neural Network
    # kl: Paper: Distilling the Knowledge in a Neural Network
    # mse: 论文：Distilling Task-Specific Knowledge from BERT into Simple Neural Networks
    # mse: Paper: Distilling Task-Specific Knowledge from BERT into Simple Neural Networks
    'distillation_method': 'ce',
    # 老师模型的类型
    # Teacher model type
    'teacher_model_type': 'Bert',
    # 学生模型的类型
    # Student model type
    'student_model_type': 'TextCNN',
    # 学生模型保存的文件夹
    # Folder to save the student model
    'checkpoints_dir': 'checkpoints/student',
    # 单独小模型训练阶段的epoch
    # Epochs for independent training of the student model
    'epoch': 100,
    # 单独小模型训练阶段的batch_size
    # Batch size for independent training of the student model
    'batch_size': 16,
    # 单独小模型训练阶段的学习率
    # Learning rate for independent training of the student model
    'learning_rate': 0.0001,
    # 单独小模型训练阶段每print_per_batch打印
    # Print frequency during independent training of the student model
    'print_per_batch': 50,
    # 单独小模型训练是否提前结束微调
    # Whether to stop training the student model early
    'is_early_stop': True,
    # 单独小模型训练阶段的patient
    # Patience for the student model's training
    'patient': 8,
    # 蒸馏损失参数平衡因子
    # Distillation loss parameter balance factor
    'alpha': 0.1,
    # 温度
    # Temperature
    'temperature': 1,
    # 小模型的名字
    # Name of the student model
    'student_model_name': 'distillation_model.pkl',
    # 老师模型的名字
    # Name of the teacher model
    'teacher_model_name': 'finetune_bert_pgd.pkl',
}

prune_configure = {
    # 剪枝的作用地
    # Location of pruning
    # vocabulary:    词表
    # vocabulary:    Vocabulary
    # transformer:   TM层
    # transformer:   Transformer layers
    # pipline:       流水线剪枝
    # pipline:       Pipeline pruning
    'model_type': 'Bert',
    # 使用配置文件
    # Use config file
    'from_config': True,
    # 剪枝的地方
    # Where to prune
    'where': 'vocabulary',
    # 最小出现次数
    # Minimum occurrence count
    'min_count': 1,
    # intermediate_size
    'intermediate_size': 2048,
    # num_attention_heads
    'num_attention_heads': 8,
    # 剪枝的迭代次数
    # Number of pruning iterations
    'iters': 4,
    # 老师模型的checkpoint文件夹
    # Teacher model checkpoint directory
    'teacher_checkpoint_dir': 'checkpoints/distillation',
    # 老师模型的名字
    # Name of the teacher model
    'teacher_model_name': 'distillation_model.bin',
    # 剪枝后模型保存的文件夹
    # Folder to save the pruned model
    'checkpoints_dir': 'checkpoints/TinyBert_PRUNE',
    # 剪枝后小模型的名字
    # Name of the pruned model
    'prune_model_name': 'prune_model.bin',
}
