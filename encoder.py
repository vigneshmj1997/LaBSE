import numpy as np
import tensorflow as tf
import bert
import tensorflow_hub as hub
import argparse
from tqdm import tqdm 


def get_model(model_url, max_seq_length):
    labse_layer = hub.KerasLayer(model_url, trainable=True)

    # Define input.
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids"
    )
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="input_mask"
    )
    segment_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), dtype=tf.int32, name="segment_ids"
    )

    # LaBSE layer.
    pooled_output, _ = labse_layer([input_word_ids, input_mask, segment_ids])

    # The embedding is l2 normalized.
    pooled_output = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(
        pooled_output
    )

    # Define model.
    return (
        tf.keras.Model(
            inputs=[input_word_ids, input_mask, segment_ids], outputs=pooled_output
        ),
        labse_layer,
    )


max_seq_length = 64
labse_model, labse_layer = get_model(
    model_url="https://tfhub.dev/google/LaBSE/1", max_seq_length=max_seq_length
)



vocab_file = labse_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = labse_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, do_lower_case)


def create_input(input_strings, tokenizer, max_seq_length):

    input_ids_all, input_mask_all, segment_ids_all = [], [], []
    for input_string in input_strings:
        # Tokenize input.
        input_tokens = ["[CLS]"] + tokenizer.tokenize(input_string) + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
        sequence_length = min(len(input_ids), max_seq_length)

        # Padding or truncation.
        if len(input_ids) >= max_seq_length:
            input_ids = input_ids[:max_seq_length]
        else:
            input_ids = input_ids + [0] * (max_seq_length - len(input_ids))

        input_mask = [1] * sequence_length + [0] * (max_seq_length - sequence_length)

        input_ids_all.append(input_ids)
        input_mask_all.append(input_mask)
        segment_ids_all.append([0] * max_seq_length)

    return np.array(input_ids_all), np.array(input_mask_all), np.array(segment_ids_all)


def encode(input_text):
    input_ids, input_mask, segment_ids = create_input(
        input_text, tokenizer, max_seq_length
    )
    return labse_model([input_ids, input_mask, segment_ids])


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--file', type=str, required=True,help="Input document file")
    parser.add_argument('--output',  type=str, default="embedding", help="Name of ouput embedding file")
    try:
        file = open(args.file).read().split()
    except:
        print("There is a problem opening {} file".format((args.file)))
   
    output = encode(file)
    output_file = open(args.output, "wb")
    output_file.tofile(output)
    print("File written in path ")


        
def __name__ == "__main__":
    main()