import datasets
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from functools import partial


class DatasetAbstract:
    def __init__(self, dataset_name: list[str], category: str):
        self.dataset_name = dataset_name
        self.metadata = {
            'domain': category
        }
    
    def _processing_data(self):
        pass
    
    @classmethod
    def get_dataset(cls, dataset_name, local_data_dir=None):
        if dataset_name in ["gsm8k"]:
            dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
            dataset = load_dataset(dataset_name, name="main")
        else:
            dataset_name = local_data_dir + dataset_name if local_data_dir is not None else dataset_name
            dataset = load_dataset(dataset_name)
        
        return dataset
    
    def get_split_dataset(self, dataset):
        print(f">> ===== After processing, Dataset has {len(dataset)} examples. =====")
        if len(dataset) > 10000:
            ds_part1, ds_part2 = train_test_split(
                dataset, test_size=0.5, shuffle=True, random_state=42
            )
            print(f">> ===== After split, Dataset1 has {len(ds_part1)} examples and Dataset2 has {len(ds_part2)} examples. =====")
            list_dataset = []
            for subset in [ds_part1, ds_part2]:
                train, test = train_test_split(
                    subset, test_size=0.2, shuffle=True, random_state=42
                )
                ds = DatasetDict({
                    "train": Dataset.from_pandas(train).remove_columns(['__index_level_0__']),
                    "test": Dataset.from_pandas(test).remove_columns(['__index_level_0__'])
                })
                list_dataset.append(ds)
            return list_dataset
                
        else:
            train, test = train_test_split(
                dataset , test_size=0.2, shuffle=True, random_state=42
            )
            ds = DatasetDict(
                {
                    "train": Dataset.from_pandas(train).remove_columns(['__index_level_0__']),
                    "test": Dataset.from_pandas(test).remove_columns(['__index_level_0__'])
                }
            )
            return [ds]

        
class GeneralDataset(DatasetAbstract):
    
    def __init__(self):
        list_dataset = ["tatsu-lab/alpaca", "vicgalle/alpaca-gpt4"]
        super().__init__(list_dataset, 'general')
        self._processing_data()
    
    def _processing_data(self):
        datasets = []
        for dataset_name in self.dataset_name:
            datasets.append(
                pd.DataFrame(super().get_dataset(dataset_name=dataset_name, local_data_dir=None)['train'])
            )
        dataset = pd.concat(datasets, ignore_index=True)
        self.list_dataset = self.get_split_dataset(dataset)
            
        

class FinanceDataset(DatasetAbstract):
    
    def __init__(self):
        list_dataset = ["gbharti/finance-alpaca", "FinGPT/fingpt-sentiment-train"]
        super().__init__(list_dataset, 'finance')
        
        self._processing_data()
    
    def _processing_data(self):
        datasets = []
        for dataset_name in self.dataset_name:
            ds = super().get_dataset(dataset_name=dataset_name, local_data_dir=None)['train']
            if dataset_name == 'gbharti/finance-alpaca':
                ds = ds.remove_columns(['text'])
            df = pd.DataFrame(ds)
            datasets.append(df)
        dataset = pd.concat(datasets, ignore_index=True)
        self.list_dataset = self.get_split_dataset(dataset)
        

class MathDataset(DatasetAbstract):
    
    def __init__(self):
        list_dataset = ["TIGER-Lab/MathInstruct", "xDAN2099/lighteval-MATH", "gsm8k"]
        super().__init__(list_dataset, 'math')
        self._processing_data()
        
    
    def get_split_dataset(self, dataset):
        dataset_train, dataset_test = dataset[0], dataset[1]
        print(f">> ===== After processing, Dataset  has {len(dataset_train)} examples. =====")
        if len(dataset_train) > 10000:
            ds_train_part1, ds_train_part2 = train_test_split(
                dataset_train, test_size=0.5, shuffle=True, random_state=42
            )
            ds_test_part1, ds_test_part2 = train_test_split(
                dataset_test, test_size=0.5, shuffle=True, random_state=42
            )
            print(f">> ===== After split, Dataset1 has {len(ds_train_part1)} examples and Dataset2 has {len(ds_train_part2)} examples. =====")
            list_dataset = []
            for i in range(2):
                ds = DatasetDict({
                    "train": Dataset.from_pandas(eval(f'ds_train_part{i+1}')).remove_columns(['__index_level_0__']), 
                    "test": Dataset.from_pandas(eval(f'ds_test_part{i+1}')).remove_columns(['__index_level_0__'])
                })
                list_dataset.append(ds)
            return list_dataset
                
        else:
            ds = DatasetDict(
                {
                    "train": Dataset.from_pandas(dataset_train).remove_columns(['__index_level_0__']),
                    "test": Dataset.from_pandas(dataset_test).remove_columns(['__index_level_0__'])
                }
            )
    
    def _processing_data(self):
        datasets_train, datasets_test = [], []
        for dataset_name in self.dataset_name:
            ds_tmp = super().get_dataset(dataset_name=dataset_name, local_data_dir=None)
            if dataset_name == 'TIGER-Lab/MathInstruct':
                df = pd.DataFrame(ds_tmp['train'])
                df = df.drop_duplicates(subset=['instruction'])
                df = df.drop(['source'], axis=1)
                df_train, df_test = train_test_split(df, test_size=0.3, shuffle=True, random_state=42)
                
            elif dataset_name == "xDAN2099/lighteval-MATH":
                ds_tmp = ds_tmp.remove_columns(['level', 'type'])
                ds_tmp = ds_tmp.rename_column("solution", "output")
                ds_tmp = ds_tmp.rename_column("problem", "instruction")
                df_train, df_test = pd.DataFrame(ds_tmp['train']), pd.DataFrame(ds_tmp['test'])
            
            elif dataset_name == 'gsm8k':
                ds_tmp = ds_tmp.rename_column("answer", "output")
                ds_tmp = ds_tmp.rename_column("question", "instruction")
                df_train, df_test = pd.DataFrame(ds_tmp['train']), pd.DataFrame(ds_tmp['test'])
            
            df_train['input'] = [''] * len(df_train)
            df_test['input'] = [''] * len(df_test)
            datasets_train.append(df_train)
            datasets_test.append(df_test)
            
        dataset_train = pd.concat(datasets_train, ignore_index=True)
        dataset_test = pd.concat(datasets_test, ignore_index=True)
        dataset = [dataset_train, dataset_test]
        self.list_dataset = self.get_split_dataset(dataset)
    

class MedicalDataset(DatasetAbstract):
    
    def __init__(self):
        list_dataset = ["medalpaca/medical_meadow_medical_flashcards"]
        super().__init__(list_dataset, 'medical')
        self._processing_data()
    
    def _processing_data(self):
        datasets = []
        for dataset_name in self.dataset_name:
            ds = super().get_dataset(dataset_name=dataset_name, local_data_dir=None)['train']
            if dataset_name == 'medalpaca/medical_meadow_medical_flashcards':
                ds = ds.remove_columns(['instruction'])
                ds = ds.rename_column("input", "instruction")
            
            df = pd.DataFrame(ds)
            df['input'] = [''] * len(df)
            datasets.append(df)
        dataset = pd.concat(datasets, ignore_index=True)
        self.list_dataset = self.get_split_dataset(dataset)
        
class CodeDataset(DatasetAbstract):
    
    def __init__(self):
        list_dataset = ["lucasmccabe-lmi/CodeAlpaca-20k", "WizardLMTeam/WizardLM_evol_instruct_70k"]
        super().__init__(list_dataset, 'code')
        self._processing_data()
    
    def _processing_data(self):
        datasets = []
        for dataset_name in self.dataset_name:
            ds = super().get_dataset(dataset_name=dataset_name, local_data_dir=None)['train']
            df = pd.DataFrame(ds)
            if dataset_name == 'WizardLMTeam/WizardLM_evol_instruct_70k':
                df['input'] = [''] * len(df)
            datasets.append(df)
        dataset = pd.concat(datasets, ignore_index=True)
        self.list_dataset = self.get_split_dataset(dataset)
        
data_domain = {
    'general': GeneralDataset().list_dataset,
    'finance': FinanceDataset().list_dataset,
    'math': MathDataset().list_dataset,
    'medical': MedicalDataset().list_dataset,
    'code': CodeDataset().list_dataset
}
        
client_id_dataset = {
    '1': data_domain['general'][0],
    '2': data_domain['general'][1],
    '3': data_domain['finance'][0],
    '4': data_domain['finance'][1],
    '5': data_domain['math'][0],
    '6': data_domain['math'][1],
    '7': data_domain['medical'][0],
    '8': data_domain['medical'][1],
    '9': data_domain['code'][0],
    '10': data_domain['code'][1],
}