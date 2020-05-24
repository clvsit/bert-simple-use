实习的这段期间，在公司做了不少 NLP 分类任务，歧义车系判断、字词重复纠错等等，期间有用过 Google 开源的 Bert 框架，也用过公司大佬制作的 T5 模型。但无论使用什么，起手 Bert 仿佛已经成为了一种“本能”（笑哭.jpg），Bert NLP 算法工程师的至交好友。

写这篇博客的目的一是为了记录先前工作的经验，此外也简单地介绍一下如何使用 Google 官方开源的 Bert 框架，因为目前很少有博客会讲如何将 Bert 训练得到的 checkpoint 转换为 savedModel，这何尝不是一种遗憾，因此我打算将这遗憾填补。对于第一次使用 Bert 框架的读者，我建议从头开始看，若已经有熟练的使用经验，只想了解如何导出模型和使用，可以直接跳到**模型导出和使用**。

## 准备工作
首先，到 GitHub 上 clone Bert 源代码。
- google-research/bert：https://github.com/google-research/bert

然后下载预训练模型，本博客的案例用的是中文数据，因此下载 [BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)（如果下载很慢，可以尝试到网盘或其他资源站点去下载）。

【数据】：案例使用的数据选择外卖情感极性评价数据集，这个数据集是我从 CSDN 的下载中找到的。
- 积极：

```
很快，好吃，味道足，量大
没有送水没有送水没有送水
非常快，态度好。
方便，快捷，味道可口，快递给力
菜味道很棒！送餐很及时！
今天师傅是不是手抖了，微辣格外辣！
送餐快,态度也特别好,辛苦啦谢谢
超级快就送到了，这么冷的天气骑士们辛苦了。谢谢你们。麻辣香锅依然很好吃。
经过上次晚了2小时，这次超级快，20分钟就送到了……
最后五分钟订的，卖家特别好接单了，谢谢。
```
- 消极：

```
菜品质量好，味道好，就是百度的问题，总是用运力原因来解释，我也不懂这是什么原因，晚了三个小时呵呵厉害吧！反正订了就退不了，只能干等……
分量还可以……就是有点没特色……下回不吃啦
没什么味道，送来的晚凉了
送餐送错，还狡辩不给补偿，送餐时间3个小时，百度送餐员更是素质卑劣，额外还要加收17元的外送费。百度客服也没用，也没有解决！
完全不值得信任，出尔反而
最差餐厅，没有之一
感觉不太好吃，价格贵。但是百度外卖的送餐人员态度很好！
没发票，乱收费，订单没到就被完成了
量很大，但是味道真的一般。等了一个多小时才送到，达到了超时赔付的时间，感觉百度超时赔付就是个摆设
头一天看同事点的三份菜就一大盆，今天点了四份菜才一小碗……差太大吧，辣椒花椒太多，非常油腻……
```

然后，我们统计所有数据的句长，其中最长的句子长度为 463。

![句长分布图](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/framework/bert/%E5%8F%A5%E9%95%BF%E5%88%86%E5%B8%83%E5%9B%BE.jpg)

可以看到句子长度的分布很不均匀，这里我们可以将 max\_seq\_length 设置成 465（加上 [CLS] 和 [SEQ]），对于长度不足 465 的句子通过 padding 填充，缺点在于占用更多的空间和花费更多的训练时间。实际上句长较长的那部分数据属于长尾数据，我们可以将其抛弃。最终得到的是句子长度小于等于 126 的数据，max\_seq\_length 设置为 128。

接着将上述两部分的数据处理成 BERT 的输入数据形式，在处理成 csv 数据格式时需要注意“,”（英文逗号）是否在原文中出现，可以看到这份数据不够“干净”，里面存在大量的错别字以及错用的标点符号，因此有两种处理方式：
- 将英文逗号替换为中文逗号，然后用英文逗号作为输入文本和标签的分隔符。
- 使用文章中没有出现的符号作为分隔符。

这里采用第二种方法，使用“&”作为文本和标签的分隔符。
```
吃的挺好的,以后还会点别的&1
送餐比之前快了不少呀，是只有我这么想么。味道很赞。&1
分量够，味道可以，送一次性手套和餐巾纸&1
不错，常客了，肘子的好吃,送货也快&1
煎饼很好吃！送餐很快！&1
皮蛋粥快咸死了，不好吃！不过包装值得表扬！&0
还行吧，因为送来时有点凉了，等的有点久&0
味儿还行，就是油太大了！&0
加了一份米饭。打开一看。两份顶一份。太坑了。2份饭不够吃。&0
沙拉恶心死了都成泥了快递竟然还没有零钱&0
2:30送到,小伙伴们,看着办吧&0
```

处理完数据之后，我们再将 bert 的代码和预训练模型 chinese\_L-12\_H-768\_A-12 放到同一个项目下，整个项目结构如下所示。

【项目结构】：
```
model/
    vocab.txt
    bert_model.ckpt.meta
    bert_model.ckpt.index
    bert_model.ckpt.data-00000-of-00001
    bert_config.json
data/
    train.csv
    dev.csv
    test.csv
bert/
    ...
output/
export/
train.sh
predict.sh
predict.py
export.sh
```

## 训练和验证
使用 BERT 框架进行模型训练非常简单，我们要做的就是修改（1）数据读取部分（2）模型配置部分。

（1）数据读取部分：找到 run\_classifier.py 文件中的 DataProcessor(object) 类，我们要做的就是继承这个类，用来处理我们自己的数据。

我们可以直接复制已有的 DataProcessor，然后修改下相应的代码。

- 修改读取文件的路径和名称。

```python
def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

def get_dev_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")

def get_test_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.csv")), "test")
```

将读取 train、dev 和 test 函数中的文件名称修改成我们的文件名称，其余都可以不用改动。

- 修改标签函数。

```python
def get_labels(self):
    return ["0", "1"]
```
因为是一个二分类任务（积极和消极），因此可以让 get_labels() 函数直接返回 0 和 1。

- 修改文件读取函数。

```python
def _read_tsv(cls, input_file, quotechar=None):
    with tf.gfile.Open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="&", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines
```
因为我们的数据是以“&”作为分隔符，因此在这需要将 `delimiter="\t"` 修改为 `delimiter="&"`。

- 修改 _create\_examples() 函数。

```python
def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        if set_type == "test":
            text_a = tokenization.convert_to_unicode(line[0])
            label = "0"
        else:
            text_a = tokenization.convert_to_unicode(line[0])
            label = tokenization.convert_to_unicode(line[1])

        if label not in ["0", "1"]:
            continue

        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
```
`_create_examples()` 函数是修改的重点，在这里处理数据中各字段的读取，因为这是一个单句子任务，因此我们只需要 text\_a 即可。

【完整代码】：
```python
class EmotionProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.csv")), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.csv")), "dev")
    
    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.csv")), "test")
            
    def get_labels(self):
        return ["0", "1"]
        
    def _read_tsv(cls, input_file, quotechar=None):
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="&", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
            
    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[0])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                label = tokenization.convert_to_unicode(line[1])
    
            if label not in ["0", "1"]:
                continue
    
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
```

最后，将新创建的 EmotionProcessor 加入到 main() 函数的 processors 中。
```python
processors = {
    "emotion": EmotionProcessor,
}
```
【注意】：这里的 emotion 需要小写，因为 BERT 在读取 task_name 时进行了小写处理。
```python
task_name = FLAGS.task_name.lower()

if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

processor = processors[task_name]()
```

（2）模型配置部分：run_classifier.py 文件是一个命令行调用脚本文件，如果是在 linux 系统上，我们可以编写 bash 脚本，在这将训练和验证一起完成。
```bash
#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=1 python bert/run_classifier.py \
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
	--output_dir=output
```

【注意】：
- 请根据自己机子的配置设置 `train_batch_size` 大小，以及 bert_config.json 中的配置内容。
- CUDA_VISIBVLE_DEVICES 指定要使用的显卡，如果只有一张显卡，设置为 0，即 `CUDA_VISIBLE_DIVICES=0`。

如果没有问题的话，我们就以 bert 默认的配置进行训练和验证。在控制台输入：
```
sh train.sh
```

等待一段时间后，直到训练完成后，我们可以在控制台中看到模型在验证集上的准确率。

![BERT train and eval](https://markdown-picture-clvsit.oss-cn-hangzhou.aliyuncs.com/nlp/framework/bert/%E6%A1%86%E6%9E%B6%E4%BD%BF%E7%94%A8%20eval.jpg)

## 模型推断
在完成模型训练后，我们可以在 output 目录下看到模型训练和验证阶段的记录和结果。

```
eval/
train.tf_record
model.ckpt-624.meta
model.ckpt-624.index
model.ckpt-624.data-00000-of-00001
model.ckpt-0.meta
model.ckpt-0.index
model.ckpt-0.data-00000-of-00001
graph.pbtxt
events.out.tfevents.xxx
eval_results.txt
eval.tf_record
checkpoint
```

其中 train.tf_record 和 eval.tf_record 是我们训练和验证数据集，eval_results.txt 是验证的结果，而 model.ckpt-624 是训练完成的模型文件，也是我们推断时指定的模型。

在项目目录下创建 predict.sh 文件。
```
#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=1 python bert/run_classifier.py \
	--task_name=Emotion \
	--do_predict=true \
	--data_dir=$DATA_DIR \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=$MODEL_DIR/model.ckpt-624 \
	--output_dir=output
```

相比 train.sh 文件，将 `do_train` 和 `do_eval` 修改为 `do_predict`，并指定 `init_checkpoint` 为我们训练完成的模型文件。如果你想将推断结果存放到其他目录下，可以修改 `output_dir` 参数值。在这，我仍然将推断的结果存储到 output 目录下，此时可以看到多了 test_results.tsv 文件。
```
0.011303517	0.98869646
0.9418804	0.058119625
0.04432816	0.9556718
0.011732221	0.98826784
0.029930793	0.9700693
0.012002373	0.98799765
0.09023312	0.90976685
0.017287388	0.9827126
0.03125599	0.968744
0.015658164	0.9843418
```

该文件记录模型对各标签的预测概率值，例如第一条消极的概率为 0.011303517，积极的概率为 0.98869646，模型认为第一条评论是积极的。查看 test.csv 文件的第一条评论“菜量很大，味道也不错，师傅速度很快，好评～”，模型的预测是正确的。

### 模型导出和使用
虽然我们可以直接使用 predict.sh 去做预测，但问题是我们需要将预测的数据做成 csv 文件，然后启动 predict.sh，能不能做成函数的形式，把输入传给一个函数，然后得到相应的结果？当然可以！

首先，我们需要在 run\_classifier.py 文件中新增导出的代码。
- 在代码的 flags 区域加上 `export_dir` 和 `do_export`。

```python
flags.DEFINE_string(
    "export_dir", None,
    "The dir where the exported model will be written.")

flags.DEFINE_bool(
    "do_export", False, "Whether to export the model.")
```

- 然后创建 `serving_input_fn()` 函数。

```python
def serving_input_fn():
    input_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_ids')
    input_mask = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='input_mask')
    segment_ids = tf.placeholder(tf.int32, [None, FLAGS.max_seq_length], name='segment_ids')

    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        'label_ids': tf.constant(0, tf.int32),
        'input_ids': input_ids,
        'input_mask': input_mask,
        'segment_ids': segment_ids
    })()
    return input_fn
```

- 接着在 main() 函数中修改如下几处代码。

```python
if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_export:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
```

- 最后，在 main() 函数的尾部加上导出相关的代码。

```python
if FLAGS.do_export:
    estimator._export_to_tpu = False
    estimator.export_savedmodel(FLAGS.export_dir, serving_input_fn)
```

完成代码的修改后，接着把训练好的模型转换成 savedModel 形式，开始编写 export.sh 文件。
```
#! /bin/bash
export MODEL_DIR=model
export DATA_DIR=data

CUDA_VISIBLE_DEVICES=1 python bert/run_classifier.py \
	--task_name=Emotion \
	--do_export=true \
	--data_dir=$DATA_DIR \
	--vocab_file=$MODEL_DIR/vocab.txt \
	--bert_config_file=$MODEL_DIR/bert_config.json \
	--init_checkpoint=$MODEL_DIR/model.ckpt-624 \
	--output_dir=output
	--export_dir=export
```

运行 `sh export.sh` 命令，等待片刻后，看到控制台输出：
```
SavedModel written to: exported/temp-b'1590300832'/saved_model.pb
```

export 目录下多了 temp-b'1590300779' 和 1590300832 的两个目录，其中 1590300832 是导出的 saveModel 以时间戳命名。

最后，让我们编写一个简单的 predict.py 脚本文件。
```python
import tensorflow as tf
from bert import tokenization


def convert_single_example(query, label_list, max_seq_length, tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(query)

    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return {
        "input_ids": input_ids,
        "segment_ids": segment_ids,
        "input_mask": input_mask
    }


if __name__ == '__main__':
    label_list = [0, 1]
    predict_fn = tf.contrib.predictor.from_saved_model("exported/1590300832/")
    tokenizer = tokenization.FullTokenizer(vocab_file="model/vocab.txt", do_lower_case=True)
    feature = convert_single_example("菜量很大，味道也不错，师傅速度很快，好评～", label_list, 128, tokenizer)
    prediction = predict_fn({
        "input_ids": [feature['input_ids']],
        "segment_ids": [feature['segment_ids']],
        "input_mask": [feature['input_mask']]
    })
    probabilities = prediction["probabilities"]
    label = label_list[probabilities.argmax()]
    print(probabilities)
    print(label)

```

其中，`convert_single_example()` 函数可以直接从 `run_classifier.py` 文件中拷贝。我们要做的实际上就是读取 saveModel 文件以及整理输入数据格式。
- 读取 saveModel：注意替换成自己模型的名称哦。

```python
predict_fn = tf.contrib.predictor.from_saved_model("exported/1590300832/")
```
- 整理输入数据格式。

```python
tokenizer = tokenization.FullTokenizer(vocab_file="model/vocab.txt", do_lower_case=True)
feature = convert_single_example("菜量很大，味道也不错，师傅速度很快，好评～", label_list, 128, tokenizer)
```
- 将输入数据传给预测函数，得到预测结果。

```python
prediction = predict_fn({
    "input_ids": [feature['input_ids']],
    "segment_ids": [feature['segment_ids']],
    "input_mask": [feature['input_mask']]
})
```

至此，关于 Bert 框架训练、验证、推断和导出的简单说明告一段落，如有错误请各位读者指出，不胜感激。