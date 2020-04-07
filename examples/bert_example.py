from text_classifier.bert_text_classifier import BertTextClassifier
# classifier = FlairTextClassifier("apollo/models/sc.pt")
classifier = BertTextClassifier(model_name_or_path="models/bert_model/")

labels = classifier.process('dễ sử dụng hơn bể này ko ạ')
print(labels)

# from flair.data_fetcher import NLPTaskDataFetcher
# corpus = NLPTaskDataFetcher.load_classification_corpus('dataset/eot_v1.4',
#                                                        dev_file='valid_v1.4_fasttext.csv',
#                                                        train_file='train_v1.4_fasttext.csv'
#                                                        , test_file='test_v1.4_fasttext.csv')
# corpus.train.__len__()
# classifier = FlairTextClassifier()
# classifier.train(corpus)
