from transformers import *
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from keras.preprocessing.sequence import pad_sequences
from utils import normalize_text, flat_accuracy

import tempfile
import pandas as pd
import numpy as np
import random
import os
import time


class BertTextClassifier:
    """text classify component"""

    def __init__(self, model_name_or_path: str = None, batchsize=32, learning_rate=2e-5, adam_epsilon=1e-8, epochs=2,
                 seed=42, MODELS=(BertForSequenceClassification, BertTokenizer, 'bert-base-multilingual-cased'),
                 max_len=128, num_label=2, gpu=False, normalization=True, output_dir="./models/", vocab=None):
        self.batch_size = batchsize
        self.learning_rate = learning_rate
        self.adam_epsilon = adam_epsilon
        self.epochs = epochs
        self.seed = seed
        self.max_len = max_len
        self.num_label = num_label
        self.MODELS = MODELS
        self.gpu = gpu
        self.normalization = normalization
        self.output_dir = output_dir
        self.vocab = vocab  # path to vocab file

        if model_name_or_path is not None:
            self.model = self.MODELS[0].from_pretrained(model_name_or_path)
            self.tokenizer = self.MODELS[1].from_pretrained(model_name_or_path, do_lower_case=False)
        else:
            # Load BertForSequenceClassification, the pre-trained BERT model with a single
            # linear classification layer on top.
            self.model = self.MODELS[0].from_pretrained(
                self.MODELS[2],  # Use the 12-layer BERT model, with an uncased vocab.
                num_labels=self.num_label,  # The number of output labels--2 for binary classification.
                # You can increase this for multi-class tasks.
                output_attentions=False,  # Whether the model returns attentions weights.
                output_hidden_states=False,  # Whether the model returns all hidden-states.
            )

            # Load pre-train tokenizer
            if self.vocab is None:
                self.tokenizer = self.MODELS[1].from_pretrained(self.MODELS[2], do_lower_case=False)
            else:
                self.tokenizer = self.MODELS[1](vocab_file=self.vocab, do_lower_case=False)

        # Set divice
        if self.gpu:
            # Tell PyTorch to use the GPU.
            torch.cuda.set_device(1)
            self.device = torch.device("cuda")

            self.model.cuda()
            # Tell PyTorch to use the GPU.

            # print('There are %d GPU(s) available.' % torch.cuda.device_count())
            #
            # print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            # print('There are 0 GPU(s) available.')
            #
            # print('We will use the CPU.')

        self.tempdir = tempfile.mkdtemp()

    def train(self, data_path, valid_path):
        """train the model"""

        # Load data
        train_data = pd.read_csv(data_path)
        valid_data = pd.read_csv(valid_path)

        # Normalization
        if self.normalization:
            train_data = self._data_normalization(train_data)
            valid_data = self._data_normalization(valid_data)

        # Create data set
        dataset = self._create_dataset(train_data)
        validset = self._create_dataset(valid_data)

        # Train model
        self._train_model(train_dataset=dataset, valid_dataset=validset)

        # Save model
        self.persist()

    def process(self, text: str):
        """predict type of text
        ex: 0-not eot, 1-eot
        """
        # normalize text
        sentence = normalize_text(text)
        # print("text normalized: ", sentence)
        # creat input for model

        self.model.eval()

        # compute output score
        input_ids = torch.tensor(
            self.tokenizer.encode(sentence, add_special_tokens=True, max_length=64, pad_to_max_length=True,
                                  return_attention_mask=True)).unsqueeze(0)
        input_ids = input_ids.to(self.device)
        outputs = self.model(input_ids)
        logits = outputs[0]
        confidence = torch.softmax(logits, dim=1).tolist()[0]
        predict = [{'sentence': text, 'label': np.argmax(confidence), 'confidence': np.max(confidence)}]
        return predict

    def load(self, model_path="albert-large-v2"):
        """
        Load model is trained
        Default is bert-base-multilingual-cased
        """

        self.model = self.MODELS[0].from_pretrained(model_path)
        if self.gpu:
            self.model.cuda()
        self.tokenizer = self.MODELS[1].from_pretrained(model_path, do_lower_case=False)

    def persist(self, model_file=None, model_dir=None):
        # Create output directory if needed to save model
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        print("Saving model to %s" % self.output_dir)

    def _fetch_model(self, model_file, model_dir):
        try:
            logger.info("Download file: %s into %s", model_file, model_dir)
        except Exception as e:
            logger.exception("Download model exception:")

    def evaluate(self, data_path):
        """Solve accuracy evaluate"""

        testdata = pd.read_csv(data_path)

        # normalize data
        testdata = self._data_normalization(testdata)

        # creat input for model
        dataset = self._create_dataset(testdata)

        # solve
        return self._evaluate(dataset)

    def _evaluate(self, dataset):

        self.model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        true_label = []
        predict_label = []
        list_label_predict = []
        list_confidence = []

        for batch in dataset:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            logits = outputs[0]

            # Compute confidence
            confidence = torch.softmax(logits, dim=1).tolist()
            predict = {'label': np.argmax(confidence, axis=1), 'confidence': np.max(confidence, axis=1)}
            list_confidence += list(np.max(confidence, axis=1))
            list_label_predict += list(np.argmax(confidence, axis=1))

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Append to predict lable and true label
            predict_label.append(logits)
            true_label.append(label_ids)

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy

            # Track the number of batches
            nb_eval_steps += 1

            # Report the final accuracy for this validation run.
        #         print(" Accuracy: {0:.4f}".format(eval_accuracy / nb_eval_steps))
        return (true_label, predict_label, round(eval_accuracy / nb_eval_steps, 4), list_label_predict, list_confidence)

    def _train_model(self, train_dataset, valid_dataset):
        """train the model"""

        # Tell pytorch to run this model on the GPU.
        # model.cuda()

        # Prepare optimizer and schedule (linear warmup and decay)
        # using AdamW optimizer in huggingface library
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            eps=self.adam_epsilon
        )

        total_steps = len(train_dataset) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # Set seed
        random.seed(self.seed)
        np.random.seed((self.seed))
        torch.manual_seed(self.seed)
        if self.gpu:
            torch.cuda.manual_seed_all(self.seed)

        # Store the average loss after each epoch
        loss_values = []

        # train
        for epoch_i in range(0, self.epochs):
            print("===== Epoch {0} / {1} =====".format(epoch_i, self.epochs))
            print("Training...")

            # time start epochs
            t0 = time.time()

            # reset total loss for this epoch
            total_loss = 0
            # Tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            self.model.train()

            for step, batch in enumerate(train_dataset):
                # progeress update every 50 batches
                if step % 50 == 0 and not step == 0:
                    time_epoch = str(round(time.time() - t0, 4))

                    # report progress
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataset), time_epoch))

                batch_input_ids = batch[0].to(self.device)
                batch_input_mask = batch[1].to(self.device)
                batch_labels = batch[2].to(self.device)

                self.model.zero_grad()

                outputs = self.model(
                    input_ids=batch_input_ids,
                    token_type_ids=None,
                    attention_mask=batch_input_mask,
                    labels=batch_labels
                )

                # The call to `model` always returns a tuple, so we need to pull the
                # loss value out of the tuple.
                loss = outputs[0]
                logits = outputs[1]

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = batch_labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences.
                tmp_eval_accuracy = flat_accuracy(logits, label_ids)

                # Accumulate the total accuracy.
                eval_accuracy += tmp_eval_accuracy

                # Track the number of batches
                nb_eval_steps += 1

                # Accumulate the training loss over all of the batches so that we can
                # calculate the average loss at the end. `loss` is a Tensor containing a
                # single value; the `.item()` function just returns the Python value
                # from the tensor.
                total_loss += loss.item()

                # Perform a backward pass to calculate the gradients.
                loss.backward()

                # Clip the norm of the gradients to 1.0.
                # This is to help prevent the "exploding gradients" problem.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient.
                # The optimizer dictates the "update rule"--how the parameters are
                # modified based on their gradients, the learning rate, etc.
                optimizer.step()

                # Update the learning rate.
                scheduler.step()

            # Calculate the average loss over the training data.
            avg_train_loss = round(total_loss / len(train_dataset), 4)

            # Accuracy
            avg_train_acc = round(eval_accuracy / nb_eval_steps, 4)

            avg_valid_acc = self._evaluate(valid_dataset)
            print(
                " train_acc: {0} , train_loss: {1} ----- valid_acc: {2} , valid_loss: {3} ----- traning epoch time: {4}".format(
                    avg_train_acc, avg_train_loss, avg_valid_acc[2], 'None', str(round(time.time() - t0, 4))
                ))

            # Store the loss value for plotting the learning curve.
            loss_values.append(avg_train_loss)

        #             print("  Average training loss: {0:.4f}".format(avg_train_loss))
        #             print("  Training epoch took: {:}".format(str(time.time() - t0)))

    def _create_dataset(self, dataset):
        """Create dataset for model"""

        # create dataset for predict sentence
        if isinstance(dataset, list):
            sentences = dataset
            labels = np.array([0])
        else:
            # Load dataset
            sentences = dataset.text.values
            labels = dataset.label.values

        ## tokenizer with my vocab
        # tokenizer =

        # Sentences to ids
        ## Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        ## For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = self.tokenizer.encode(
                sent,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'

                # This function also supports truncation and conversion
                # to pytorch tensors, but we need to do padding, so we
                # can't use these features :( .
                # max_length = 128,          # Truncate all sentences.
                # return_tensors = 'pt',     # Return pytorch tensors.
            )

            ## Add the encoded sentence to the list.
            input_ids.append(encoded_sent)

        # Padding and Truncating
        input_ids = pad_sequences(input_ids, maxlen=self.max_len, dtype="long",
                                  value=0, truncating="post", padding="post")

        # Attention Masks
        ## Create attention masks
        attention_masks = []

        ## For each sentence...
        for sent in input_ids:
            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id > 0) for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)

        # Convert all inputs and labels into torch tensors, the required datatype for model
        input_ids = torch.tensor(input_ids)
        attention_masks = torch.tensor(attention_masks)
        labels = torch.tensor(labels)

        # Create the DataLoader for training set
        # Create the DataLoader for our training set.
        train_data = TensorDataset(input_ids, attention_masks, labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        return train_dataloader

    def _data_normalization(self, data):
        """
        pre-processing data
        Data format: 2 columns text (type:str), label (type:int)
        """
        # create columns text normalize
        data.columns = ['text', 'label']
        data['text'] = data.text.map(normalize_text)
        return data[['text', 'label']]
