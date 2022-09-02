import re
import io
import pandas as pd

from deepmultilingualpunctuation import PunctuationModel
from typing import List, Optional
from tqdm import tqdm

tqdm.pandas()

class NerDataset():
    def __init__(self, filepath: str, cols: Optional[List[str]]) -> None:
        '''
        Currenly, limited to prepare an already downloaded .csv from elastic. In the future, the elastic connector
        will be implemented to pull data at will. 
        '''
        self.punct_model = PunctuationModel(model="oliverguhr/fullstop-dutch-punctuation-prediction")
        self.dataset = pd.read_csv(filepath,usecols=cols,delimiter=";")


    def prepare(self) -> None:
        
        self.dataset[self.dataset['dam.my_problem_is'] != '-'].reset_index(drop=True)
        self.dataset.rename(columns={'dam.my_problem_is': 'my_problem_is'}, inplace=True)

        self.dataset['my_problem_is_punct'] = self.dataset.progress_apply(lambda x: self.restore_punctuation(x.my_problem_is), axis=1)

        self.cleaned_texts = self.dataset.my_problem_is_punct.values

        self.cleaned_texts = [re.sub("[\(\[].*?[\)\]]", "", str(conversation)) for conversation in self.cleaned_texts]
        self.cleaned_texts = [''.join(conversation.lstrip().splitlines()) for conversation in self.cleaned_texts]

        assert len(self.dataset) == len(self.cleaned_texts)

        return self

    def export(self,output: str) -> None:
        
        with open(output, "w",encoding="utf-8") as outfile:
            outfile.write("".join(self.cleaned_texts))
        print("Dataset created...")

        return self

    def restore_punctuation(self, x: str) -> str:
        sentences = x.split("\n")
        reconstruction = []
        for sent in sentences:
            sent = re.sub("\.\.\.", "", sent)
            result = self.punct_model.restore_punctuation(sent)
            # print(result)
            reconstruction.append(result+"\n")
            # reconstruction.append(sent+"\n")
        string = ''.join(reconstruction)

        return string
