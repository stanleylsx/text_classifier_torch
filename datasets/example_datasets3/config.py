# -*- coding: utf-8 -*-
# @Time : 2021/07/15 21:03 下午
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : config.py
# @Software: PyCharm

# 模式
# train_classifier:     训练分类器
# interactive_predict:  交互模式
# test:                 跑测试集
# convert_onnx:         将torch模型保存onnx文件
# show_model_info:      打印模型参数
mode = 'train_classifier'

# 阶段
# finetune:             微调预训练模型
# train_small_model:    单独训练小模型
# distillation:         蒸馏
# prune:                模型剪枝
stage = 'finetune'

# 支持的模型
support_model = {
    'pretrained_types': ['Bert', 'DistillBert', 'RoBerta', 'ALBert', 'XLNet', 'Electra', 'MiniLM', 'DeBertaV3', 'XLM-RoBERTa'],
    'ordinary_types': ['FastText', 'TextCNN', 'TextRNN', 'TextRCNN', 'Transformer']
}

# 使用GPU设备
use_cuda = True
cuda_device = -1

train_configure = {
    # 训练数据集
    'train_file': 'datasets/example_datasets3/train_dataset.csv',
    # 验证数据集
    'val_file': '',
    # 是否多标签分类
    'multilabel': True,
    # 使用交叉验证
    'kfold': False,
    'fold_splits': 5,
    # 没有验证集时，从训练集抽取验证集比例
    'validation_rate': 0.15,
    # 测试数据集
    'test_file': 'datasets/example_datasets3/test_dataset.csv',
    # token粒度,token选择字粒度的时候，词嵌入无效
    # 词粒度:'word'
    # 字粒度:'char'
    'token_level': 'char',
    # 去停用词，仅限非预训练微调使用，字粒度不建议用
    'stop_words': False,
    'stop_words_file': 'datasets/example_datasets2/stop_words.txt',
    # 是否去掉特殊字符
    'remove_special': True,
    # 存放词表的地方，使用预训练模型的时候留空
    'token_file': 'datasets/example_datasets3/char-token2id',
    # 类别列表
    'classes': ['组织关系-辞/离职',
                '司法行为-拘捕',
                '人生-死亡',
                '人生-庆生',
                '竞赛行为-退役',
                '灾害/意外-洪灾',
                '组织关系-停职',
                '组织关系-解散',
                '财经/交易-加息',
                '司法行为-立案',
                '人生-失联',
                '组织行为-开幕',
                '组织行为-闭幕',
                '产品行为-召回',
                '司法行为-约谈',
                '竞赛行为-禁赛',
                '灾害/意外-地震',
                '灾害/意外-坠机',
                '组织关系-解雇',
                '司法行为-开庭',
                '司法行为-起诉',
                '交往-点赞',
                '产品行为-获奖',
                '组织关系-解约',
                '财经/交易-出售/收购',
                '产品行为-上映',
                '司法行为-罚款',
                '组织关系-退出',
                '交往-会见',
                '财经/交易-降息',
                '财经/交易-跌停',
                '人生-离婚',
                '财经/交易-上市',
                '交往-感谢',
                '灾害/意外-车祸',
                '产品行为-下架',
                '人生-怀孕',
                '人生-求婚',
                '组织行为-游行',
                '组织关系-裁员',
                '司法行为-举报',
                '财经/交易-涨停',
                '财经/交易-融资',
                '竞赛行为-晋级',
                '灾害/意外-袭击',
                '产品行为-发布',
                '财经/交易-涨价',
                '交往-道歉',
                '竞赛行为-胜负',
                '人生-结婚',
                '人生-婚礼',
                '灾害/意外-坍/垮塌',
                '竞赛行为-退赛',
                '交往-探班',
                '组织关系-加盟',
                '竞赛行为-夺冠',
                '财经/交易-降价',
                '人生-出轨',
                '组织行为-罢工',
                '灾害/意外-爆炸',
                '人生-订婚',
                '司法行为-入狱',
                '人生-产子/女',
                '灾害/意外-起火',
                '人生-分手'],
    # 模型保存的文件夹
    'checkpoints_dir': 'checkpoints/example3',
    # 句子最大长度
    'max_sequence_length': 100,
    # 遗忘率
    'dropout_rate': 0.5,
    # 若为二分类则使用binary
    # 多分类使用micro或macro
    # 多标签分类用samples
    'metrics_average': 'samples',
    # 多标签分类使用苏神的损失函数
    'use_multilabel_categorical_cross_entropy': True,
    # 类别样本比例失衡的时候可以考虑使用
    'use_focal_loss': False,
    # focal loss的各个标签权重
    'weight': None,
    # 使用Poly Loss
    'use_poly_loss': False,
    # 使用标签平滑
    'use_label_smoothing': True,
    'smooth_factor': 0.1,
    # 使用对抗学习
    'use_gan': True,
    # fgsm:Fast Gradient Sign Method
    # fgm:Fast Gradient Method
    # pgd:Projected Gradient Descent
    # freelb:Free Large Batch Adversarial Training
    # awp: Weighted Adversarial Perturbation
    'gan_method': 'fgm',
    # 对抗次数
    'attack_round': 3,
    # 使用对比学习，不推荐和对抗方法一起使用，效率慢收益不大
    'use_r_drop': False,
    # 使用Multisample Dropout
    # 使用Multisample Dropout后dropout会失效
    'multisample_dropout': True,
    'dropout_round': 5,
    # 随机种子
    'seed': 3407,
    # 预训练模型是否前置加入Noisy
    'noisy_tune': False,
    'noise_lambda': 0.12,
    # 是否进行warmup
    'warmup': True,
    # warmup方法，可选：linear、cosine
    'scheduler_type': 'linear',
    # warmup步数，-1自动推断为总步数的0.1
    'num_warmup_steps': -1,
    # 是否进行随机权重平均swa
    'swa': False,
    'swa_start_step': 5000,
    'swa_lr': 1e-6,
    # 每个多久平均一次
    'anneal_epochs': 1,
    # 使用EMA
    'ema': True,
    # 优化器选择
    'optimizer': 'AdamW',
    # 执行权重初始化，仅限于非微调
    'init_network': False,
    # 权重初始化方式，可选：xavier、kaiming、normal
    'init_network_method': 'xavier',
    # fp16混合精度训练，仅在Cuda支持下使用
    'use_fp16': False,

    # 预训练模型类型
    'f_model_type': 'Bert',
    # 预训练模型细分类
    'ptm': 'bert-base-chinese',
    # 微调阶段的epoch
    'f_epoch': 30,
    # 微调阶段的batch_size
    'f_batch_size': 32,
    # 微调阶段的学习率
    'f_learning_rate': 4e-5,
    # 微调阶段每print_per_batch打印
    'f_print_per_batch': 50,
    # 是否提前结束微调
    'f_is_early_stop': True,
    # 微调阶段的patient
    'f_patient': 3,
    # Bert模型的名字
    'f_model_name': 'torch.bin',

    # 小模型的类型
    's_model_type': 'TextCNN',
    # Embedding向量维度
    'embedding_dim': 300,
    # TextCNN卷积核的个数
    'num_filters': 64,
    # TextRNN单个RNN的隐藏神经元个数
    # Transformer中Feed-Forward层的隐藏层维度
    'hidden_dim': 1024,
    # 使用attention
    'use_attention': False,
    # 编码器个数(使用Transformer需要设定)
    'encoder_num': 1,
    # 多头注意力的个数(使用Transformer需要设定)
    'head_num': 4,
    # 单独小模型训练阶段的epoch
    's_epoch': 100,
    # 单独小模型训练阶段的batch_size
    's_batch_size': 32,
    # 单独小模型训练阶段的学习率
    's_learning_rate': 0.001,
    # 单独小模型训练阶段每print_per_batch打印
    's_print_per_batch': 50,
    # 单独小模型训练是否提前结束微调
    's_is_early_stop': True,
    # 单独小模型训练阶段的patient
    's_patient': 8,
    # 小模型的名字
    's_model_name': 'torch.bin',
}

distill_configure = {
    # 自蒸馏
    'self_distillation': True,
    # 蒸馏方法
    # ce: 最原始的蒸馏办法
    # kl: 论文：Distilling the Knowledge in a Neural Network
    # mse: 论文：Distilling Task-Specific Knowledge from BERT into Simple Neural Networks
    'distillation_method': 'ce',
    # 老师模型的类型
    'teacher_model_type': 'Bert',
    # 学生模型的类型
    'student_model_type': 'Bert',
    # 学生模型保存的文件夹
    'checkpoints_dir': 'checkpoints/student',
    # 单独小模型训练阶段的epoch
    'epoch': 100,
    # 单独小模型训练阶段的batch_size
    'batch_size': 32,
    # 单独小模型训练阶段的学习率
    'learning_rate': 0.0001,
    # 单独小模型训练阶段每print_per_batch打印
    'print_per_batch': 50,
    # 单独小模型训练是否提前结束微调
    'is_early_stop': True,
    # 单独小模型训练阶段的patient
    'patient': 2,
    # 蒸馏损失参数平衡因子
    'alpha': 0.1,
    # 温度
    'temperature': 4,
    # 小模型的名字
    'student_model_name': 'distillation_model.pkl',
    # 老师模型的名字
    'teacher_model_name': 'finetune_bert_pgd.pkl',
    'distill_mlm_config': {
        'attention_probs_dropout_prob': 0.1,
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'hidden_size': 768,
        'initializer_range': 0.02,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'num_attention_heads': 12,
        'num_hidden_layers': 3,
        'type_vocab_size': 2,
        'vocab_size': 21128
    },
    # 自蒸馏的时候层级映射
    'intermediate_matches': [
        {'layer_T': 0, 'layer_S': 0, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1},
        {'layer_T': 8, 'layer_S': 2, 'feature': 'hidden', 'loss': 'hidden_mse', 'weight': 1}
    ]
}

prune_configure = {
    # 剪枝的作用地
    # vocabulary:    词表
    # transformer:   TM层
    # pipline:       流水线剪枝
    'model_type': 'Bert',
    'from_config': True,
    'where': 'vocabulary',
    'min_count': 1,
    'intermediate_size': 2048,
    'num_attention_heads': 8,
    'iters': 4,
    'teacher_checkpoint_dir': 'checkpoints/distillation',
    'teacher_model_name': 'distillation_model.bin',
    # 剪枝后模型保存的文件夹
    'checkpoints_dir': 'checkpoints/TinyBert_PRUNE'
}
