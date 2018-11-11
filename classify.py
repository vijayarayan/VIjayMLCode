import pandas as pd
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding

output_dir = "spacy-model"
split = 0.8     # 80/20 train/test split
n_iter = 20     # Epoch count

def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

# load empty spaCy model
nlp = spacy.blank('en')

# add the text classifier to the pipeline
textcat = nlp.create_pipe('textcat')
nlp.add_pipe(textcat, last=True)

# add label to text classifier
textcat.add_label('HAMSPAM')

# load data
df = pd.read_csv('/Users/vmisra/eclipse-workspace/MLProject/data/vmdata/sms-train.csv')
# shuffle data
df = df.sample(frac=1).reset_index(drop=True)
# split dataframe into lists
texts = df['SMS'].tolist()
labels = df['HAM-SPAM'].tolist()
# convert "Ham"/"Spam" to boolean label
cats = [{'HAMSPAM': bool(label == "Ham")} for label in labels]
# calculate size of training set
split = int(len(texts) * split)
# split into train and test sets
train_texts = texts[:split]
train_cats = cats[:split]
dev_texts = texts[split:]
dev_cats = cats[split:]
print("Using {} examples ({} training, {} evaluation)" .format(len(texts), len(train_texts), len(dev_texts)))
train_data = list(zip(train_texts, [{'cats': cats} for cats in train_cats]))

# get names of other pipes to disable them during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print("Training the model...")
    print('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
    for i in range(n_iter):
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
        with textcat.model.use_params(optimizer.averages):
            # evaluate on the dev data split off in load_data()
            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats)
        print('{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}'
                .format((i + 1), losses['textcat'], scores['textcat_p'],
                        scores['textcat_r'], scores['textcat_f']))

# test the trained model
test_text = "Dont know you bring some food"
doc = nlp(test_text)
print(test_text, doc.cats)

#save the model
output_dir = Path(output_dir)
if not output_dir.exists():
    output_dir.mkdir()
nlp.to_disk(output_dir)
print("Saved model to", output_dir)

# test the saved model
print("Loading from", output_dir)
nlp2 = spacy.load(output_dir)
doc2 = nlp2(test_text)
print(test_text, doc2.cats)
