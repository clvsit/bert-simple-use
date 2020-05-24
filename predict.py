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
