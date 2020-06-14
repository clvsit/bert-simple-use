预训练语言模型，然后对下游任务进行微调已成为自然语言处理的新范例。预训练语言模型（PLM），例如 BERT、XLNet、RoBERTa 在许多 NLP 任务中都取得了巨大的成功。但是，PLM 通常具有大量的参数，并且需要较长的推断时间，因此很难在移动设备上进行部署。此外，最近的研究也证明了 PLM 中存在冗余。因此，在保持性能的同时减少 PLM 的计算开销和模型存储至关重要且可行。

本篇博客主要讲述论文《Distilling the Knowledge in a Neural Network》以及如何将论文中的蒸馏方法应用到 Google 官方 Bert 框架中。首先介绍论文，然后讲述使用方法。

## Distilling the Knowledge in a Neural Network
作者在这篇论文中提到了“知识蒸馏”的概念，将大模型中的知识或模式提炼到小模型中，或者说让小模型去学习大模型中的知识或模式，因为这个过程类似于工业上的蒸馏，故而命名为“知识蒸馏”。

小模型因为规模小（参数少）泛化能力相比大模型要弱，但小模型占用的存储空间更少，且推断速度更快。我们希望保留小模型这些优点的同时尽可能地提高泛化能力，该怎么做呢？知识蒸馏即是解决这个问题的方案之一——通过知识蒸馏让小模型拥有等于或近似大模型的精度。

作者在论文中提出了师生框架，将大模型当做老师模型，小模型当做学生模型，让小模型去学习大模型的预测分布。怎么理解这句话呢？仍然以外卖评价情感极性任务（消极为 0，积极为 1）为例（具体内容可参考[《Google Bert 框架训练、验证、推断和导出简单说明》](https://blog.csdn.net/weixin_43378396/article/details/106314937)），我们先让 Bert_Base 模型（12 层）在训练数据上做 fine tune，然后进行预测，这样就能得到对于每一个评论的概率分布，例如：
```
方便，快捷，味道可口，快递给力 0.001 0.999
不好吃，送得还慢，差评！       0.999 0.001
```

此时，我们让小模型例如 Bert_Small 模型（4 层）去拟合大模型的概率分布，而不是我们标注的标签。也就是说，原本我们输给小模型的是 ground-truth 转换为 one-hot 向量，例如 [0 1] 这样的 hard label，而现在输给小模型的是大模型的概率分布，例如 [0.001 0.999] 的 soft label。

#### soft label 如何获得呢？
soft label 是大模型预测的概率分布，但大模型在训练时使用的是 hard label，而交叉熵损失函数会让模型的预测极力向正确的标签靠近，例如对于数据：
```
方便，快捷，味道可口，快递给力 1
```
随着训练过程的进行，模型预测为分类 1 的概率会逐渐逼近 1，假设最终得到 0.0001 0.9999。实际上与 one-hot 向量的区别并不大。为此，作者提出了 soft softmax 函数的概念。
```math
soft-softmax(z_i) = \frac{exp(z_i / T)}{\sum_j exp(z_j / T)} 
```

相当于对输入给 softmax() 的值都除以了一个常数 T，T 是 Temperature 温度的缩写，目的在于平滑概率分布。当 T = 1 时，soft-softmax 函数即为 softmax 函数。

![exp T 分布](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/dl/distill/exp%20T.jpg)
- 红线：exp(x)
- 蓝线：exp(x/5)
- 绿线：exp(x/10)
- 紫线：exp(x/20)

通过增大 T，我们能够将原本 0.0001 0.9999 的概率分布平滑成 0.3 0.7，这么做有什么好处呢？我理解这能更好地揭示数据结构间的相似性。假设我们现在要做图像识别，根据图片的主体将其划分为猫、狗和汽车。现在有一张猫的图片，模型的预测概率为：

T | 猫 | 狗 | 车
---|---|---|---
1 | 0.9990 | 0.009 | 0.001
10 | 0.7000 | 0.2900 | 0.010

在原有的标签体系下，我们是无法知道分类之间的相似性，例如猫的图片我们只会标注为“猫”，体现在代码中即为 1 0 0，但实际上猫与狗之间存在许多的相似处，例如都能“说话”，而车不能。通过 T，能够将隐藏在概率分布中的数据结构相似性知识显露出来，从而让模型能够学到更多的知识。

#### 蒸馏目标
作者除了提出 soft-softmax 函数外，还创建了模型蒸馏的训练目标（损失函数）。
> When the correct labels are known for all or some of the transfer set, this method can be significantly
improved by also training the distilled model to produce the correct labels. One way to do this is
to use the correct labels to modify the soft targets, but we found that a better way is to simply use
a weighted average of two different objective functions.

实际上，我们可以直接让小模型去学习大模型的 soft label，但作者认为大模型的预测不一定准确，需要通过正确的标签来修正大模型输出的 soft label。作者在实验上发现简单地使用交叉熵损失和蒸馏损失的加权平均能取得更好的结果，因此最终的损失函数为两部分的加权和。

```math
L_{KD} = (1 - \alpha)H(y, \sigma(z^S)) + \alpha T^2 H(\sigma(z^T / T), \sigma(z^S / T))
```

其中，H() 表示交叉熵损失函数，`$\sigma()$` 表示 softmax 函数，T 是温度参数，y 表示数据的真实标签，上标 T 和 S 分别表示教师模型和学生模型，`$\alpha$` 是权重超参数用以平衡交叉熵损失和蒸馏损失。

实际上，公式的前半段即为我们平常使用的交叉熵损失，后半段为蒸馏损失，通过超参数 `$\alpha$` 以 soft 的形式将两者连接起来。需要注意的是，在蒸馏损失中因为除了 T，在反向传播时相比一般的交叉熵损失多了 `$1/T^2$`。为了让两部分损失在同一个量级上，因此需要在蒸馏损失前乘上 `$T^2$`。

#### 蒸馏过程
> In the simplest form of distillation, knowledge is transferred to the distilled model by training it on
a transfer set and using a soft target distribution for each case in the transfer set that is produced by
using the cumbersome model with a high temperature in its softmax. The same high temperature is
used when training the distilled model, but after it has been trained it uses a temperature of 1.

- 训练时：使用较大的 T 值来训练大模型，使大模型能够产生更平滑、均匀分布的 soft label，然后小模型使用相同的 T 值来学习 soft label。
- 推断时：在实际应用中，将 T 值调整回 1，让类别概率偏向正确类别。

## 代码实现
我们在 Google 官方提供的 Bert 框架上进行修改，主要有两种处理方式：
- 先训练大模型，然后使用训练后的大模型生成 soft label，然后将 soft label 添加到小模型的输入中。这种方式需要改动较多的代码，相当于将 soft label 一路从输入传递到最终的输出。
- 先训练大模型，然后将大模型导出成 saveModel。小模型在训练过程中调用大模型，从而生成相应的 soft label。本篇博客选择这一种方式。

### 实现细节
具体的修改代码由于过多，就不一一陈列在博客中了，大家如果感兴趣可以前往 GitHub 获取，地址：https://github.com/clvsit/bert-simple-use 。

#### run\_classifier.py
首先，我们需要修改 run\_classifier.py 脚本文件。新增两个新的输入参数 temperature 和 do_distill。
- temperature：蒸馏的温度值；
- do_distill：是否要进行蒸馏。

```python
flags.DEFINE_integer("temperature", 1, "Temperature parameters in distillation operation.")
flags.DEFINE_bool("do_distill", False, "Whether to distill the model.")
```

然后，修改 create_model() 函数，如果需要蒸馏，则执行蒸馏相关的操作。
```python
with tf.variable_scope("loss"):
    if is_training:
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    if FLAGS.do_distill:
        probabilities = tf.nn.softmax(logits / FLAGS.temperature, axis=-1)
        log_probs = tf.nn.log_softmax(logits / FLAGS.temperature, axis=-1)
        per_example_loss = -FLAGS.temperature ** 2 * tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    else:
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)
```

完成代码修改后（上述代码是为了训练教师模型），编写相应的 bash 命令。
- train\_teacher\_model.sh

```bash
#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=0 python bert/run_classifier.py \
	--task_name=Emotion \
	--do_train=true \
	--do_eval=true \
	--data_dir=$DATA_DIR \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=$MODEL_DIR/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=2.0 \
	--output_dir=output/teacher \
```

然后再使用另一篇博客[Google Bert 框架训练、验证、推断和导出简单说明](https://blog.csdn.net/weixin_43378396/article/details/106314937)中介绍的模型导出方法，导出成 savedModel 模型。

#### run_classifier_distill.py
接下来编写 run_classifier_distill.py 脚本文件，该文件用来读取教师模型，并对学生模型进行训练和蒸馏。因内容过多，具体可参考 GitHub 中的代码。

同样，编写蒸馏操作的 bash 命令。
- distill.sh

```bash
#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=2 python bert/run_classifier.py \
	--task_name=Emotion \
	--do_train=true \
	--do_eval=true \
	--data_dir=$DATA_DIR \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=$MODEL_DIR/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=2.0 \
	--teacher_model=export/1591538997 \
	--temperature=10 \
	--alpha=0.5 \
	--output_dir=output/distill
```

### 蒸馏结果
模型 | 验证集精度 | 测试集精度
---|:---:|:---:
Bert_Base | - | 0.60192
Bert\_Base + FT | 0.9182692 | 0.91635
Bert\_Small + FT | 0.9110577 | 0.91186
Bert_Small + FT + Distill | 0.9166667 | 0.91090

其中 FT 表示 fine tune，从上述结果来看模型蒸馏并没有提高 Bert Small 模型的泛化能力，其中一个较大的原因是数据量不够多，并且任务较简单，Bert Small 本身就可以学得很好，此时再蒸馏意义不大。推荐在数据量较大，任务较复杂，且需要加快推断速度的场景下尝试使用模型蒸馏技术。

如有错误，麻烦指出，不胜感激。
