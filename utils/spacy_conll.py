import spacy
import io

from typing import List, Optional
from tqdm import tqdm
from spacy.tokens import DocBin

from langs import MODEL_MAP

class ConllConverter():
    def __init__(self, language: str, documents : DocBin, tag = "BIO") -> None:

        try:
            self.documents = DocBin().from_disk(documents)
            print("Annotated Data loaded...")
        except:
            raise Exception("Cannot read data")
        self.tag = tag
        try:
            self.language = MODEL_MAP.get(language)
            print("Language: {0}".format(self.language))

            self.nlp = spacy.load(self.language, disable = ['lemmatizer','ner','attribute_ruler','tagger'])
            print("Spacy pipeline launched...")
        except:
            raise Exception("Language {} not found".format(self.language))

        self.conll_data = str()


    def convert(self):
        print("Converting Docbin to list...")
        docs=list(self.documents.get_docs(self.nlp.vocab))
        print("EXTENDED DOCS {}".format(len(docs)))
        self.result = list()
        docstart = "-DOCSTART-"
        empty = "O"
        for index, doc in enumerate(tqdm(docs)):

            self.result.append("\t".join((docstart,empty)))
            self.result.append("\n")
            self.result.append("\n")
            for sent in doc.sents:
                for tok in sent:
                        label = tok.ent_iob_
                        if tok.ent_iob_ != "O":
                            label += '-' + tok.ent_type_
                        if not str(tok).isspace():

                            self.result.append("sentence " +str(index+1)  +"\t" + str(tok) + "\t" + label)
                            self.result.append("\n")

                self.result.append("\n")


        return self

    def write(self, output_path: str):

        with open(output_path, "w",encoding="utf-8") as outfile:
            outfile.write("".join(self.result))
        print("CoNLL done")

        return self


# conv = ConllConverter(language="nl", documents = "spacy_datasets/kg_ner26K_1.9.spacy").convert()
# conv.write(output_path = "annotation_1.9.26K.txt")