import csv
from pathlib import Path
import spacy

# load the pretrained model
model = Path("spacy-model")
nlp = spacy.load(model)

# open the SMSs to be classified
with open('sms-eval.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    # open the output file
    with open('answers.csv', mode='w') as answers:
        answers_writer = csv.writer(answers, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        # write headers
        answers_writer.writerow(['ID', 'HAM-SPAM'])
        # classify the SMSs
        for row in csv_reader:
            if line_count == 0:
                # first row is a header
                print(f'{", ".join(row)}')
            else:
                # classify the tweet
                index = row[0]
                tweet = row[1]
                doc = nlp(tweet)
                cat = doc.cats['HAMSPAM']
                if cat >= 0.5:
                    cat = "Ham"
                else:
                    cat = "Spam"
                print(f'{index}\t{tweet}\t{cat}')
                answers_writer.writerow([index, cat])
            line_count += 1
        print(f'Processed {line_count - 1} SMSs.')
