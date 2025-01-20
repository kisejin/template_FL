"""flowertune-llm: A Flower / FlowerTune app."""

import os
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
from datasets import load_dataset, Dataset
from tqdm import tqdm

from transformers import DataCollatorForSeq2Seq, DataCollatorWithPadding, TrainingArguments, Trainer, GenerationConfig
from transformers.integrations import WandbCallback
from torch.utils.data import DataLoader
from flwr.common import Context, ndarrays_to_parameters
from flwr.common.config import unflatten_dict
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
# from flwr.server.strategy import FedAvg
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split

from .models import get_model, get_parameters, set_parameters
from .dataset import replace_keys
from .myfedavg import FedAvg
from .data_domains import global_test_set_hete
from .make_data import Prompter, generate_and_tokenize_prompt
from .metrics import exact_match, f1, get_rouge_score

load_dotenv(".env")

os.environ["WANDB_API_KEY"] = os.getenv("WANDB_API_KEY")
os.environ["WANDB_NAME"] = os.getenv("WANDB_NAME")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"



class LLMSampleCB(WandbCallback):
    def __init__(self, trainer, test_dataset, task, num_samples=10, max_new_tokens=256, log_model="checkpoint"):
        "A CallBack to log samples a wandb.Table during training"
        super().__init__()
        # self._log_model = log_model
        self.task = task
        self.sample_dataset = test_dataset.shuffle().select(range(num_samples))
        self.model, self.tokenizer = trainer.model, trainer.tokenizer
        self.max_new_tokens = max_new_tokens
        self.gen_config = GenerationConfig.from_pretrained(trainer.model.name_or_path,
                                                           max_new_tokens=max_new_tokens)
    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(
            prompt, 
            # padding='max_length', max_length=self.max_new_tokens, 
            return_tensors='pt'
        )
        input_ids = tokenized_prompt['input_ids'].to('cuda:0')
        
        with torch.inference_mode():
            output = self.model.generate(input_ids, generation_config=self.gen_config)
        return self.tokenizer.decode(output[0][len(tokenized_prompt[0]):], skip_special_tokens=True)
    
    def samples_table(self, examples):
        "Create a wandb.Table to store the generations"
        records_table = wandb.Table(columns=["input", "prediction", "label", "task"] + list(self.gen_config.to_dict().keys()))
        for example in tqdm(examples, leave=False):
            instruction = example["instruction"]
            inputt = example["input"]
            output = example['output']
            prompt = ''
            if inputt == '':
                prompt = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Response: """
            else:
                prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Input: {inputt} ### Response:""" 
        
            generation = self.generate(prompt=prompt)
            records_table.add_data(prompt, generation, output, self.task, *list(self.gen_config.to_dict().values()))
        return records_table
        
    def on_evaluate(self, args, state, control,  **kwargs):
        "Log the wandb.Table after calling trainer.evaluate"
        super().on_evaluate(args, state, control, **kwargs)
        records_table = self.samples_table(self.sample_dataset)
        self._wandb.log({"sample_predictions":records_table})



def test_model(dataset, model, tokenizer, train_cfg, tmp_dict, sround, task):
    
    # wandb.init(
    #         project='FL@CSS25',
    #         name=f'global_eval_round_{sround}',
    #         id=f"round_{sround}",
    #         resume="allow",
    #         reinit=True,
    #         # settings=wandb.Settings(start_method="thread")
    # )
    
    def compute_metrics(pred):
        labels_ids = pred.label_ids
        labels_ids[labels_ids == -100] = 1829
        pred_ids = np.argmax(pred.predictions, axis=-1)
        
        # all unnecessary tokens are removed
        pred_str = tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True
        )
        label_str = tokenizer.batch_decode(
            labels_ids, skip_special_tokens=True
        )
        return {
            **get_rouge_score(predictions=pred_str, targets=label_str),
            **f1(predictions=pred_str, targets=label_str),
        }
    
    data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True,
    )
    
    testset = (
            dataset
            .shuffle()
            .map(
                lambda x: generate_and_tokenize_prompt(x, **tmp_dict),
                num_proc=8,
            )
    )
    
    training_arguments = TrainingArguments(**train_cfg.training_arguments)
    training_arguments.output_dir = './global_results'
    training_arguments.logging_dir='./global_logs'
    training_arguments.run_name = f'global_eval_round_{sround}_{task}'
    
    
    
    # Constuct baseline Trainer
    trainer = Trainer(
        model=model,
        eval_dataset=testset.select(range(10)),
        args=training_arguments,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )
    
      
    trainer.add_callback(LLMSampleCB(trainer, testset, task, num_samples=5, max_new_tokens=256, log_model="checkpoint"))

    # Do local training
    results = trainer.evaluate()
    
    # Extract loss, predictions, and labels

    eval_loss = results[f"eval_loss"]
    eval_metrics = {
        f'{task}_f1': results["eval_f1"],
        f'{task}_rouge1': results["eval_rouge1"],
        f'{task}_rouge2': results['eval_rouge2'],
        f'{task}_rougeL': results['eval_rougeL'],
        f'{task}_rougeLsum': results['eval_rougeLsum'],
    }

    wandb.finish()
    
    return eval_loss, eval_metrics
    
    

# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints

def get_evaluate_fn(train_cfg, model_cfg, dataset_cfg, save_every_round, total_round, total_nodes, save_path):
    """Return an evaluation function for saving global model."""

    def evaluate(server_round: int, parameters, config):
        # Save model
        total_loss, result_metric = 0, {}
        prompter = Prompter(train_cfg.prompt_template_name, train_cfg.verbose)
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model, tokenizer = get_model(model_cfg)
            set_parameters(model, parameters)
            tmp_dict = {
                "prompter": prompter,
                "seq_length": train_cfg.seq_length,
                "train_on_inputs": train_cfg.train_on_inputs,
                "tokenizer": tokenizer,
            }
            if dataset_cfg.type == 'homo':
                ds = load_dataset(dataset_cfg.name)
                _, test = train_test_split(
                    ds, test_size=0.09, shuffle=True, random_state=42
                )
                global_test_set_homo = Dataset.from_pandas(test).remove_columns(['__index_level_0__'])
                loss, metrics = test_model(global_test_set_homo, model, tokenizer, train_cfg, tmp_dict, server_round, 'homo')
                total_loss = loss
                result_metric = {'homo_f1': metrics['homo_f1']}
            else:
                (
                    list_loss, list_f1, 
                    list_rouge1, list_rouge2, 
                    list_rougeL, list_rougeLsum 
                ) = [], {}, {}, {}, {}, {}
                
                for task in ['general', 'finance', 'math', 'medical', 'code']:
                    ds = global_test_set_hete[task]
                    loss, metrics = test_model(ds, model, tokenizer, train_cfg, tmp_dict, server_round, task)
                    list_loss.append(loss)
                    
                    list_f1[f'{task}_f1'] = metrics[f'{task}_f1']
                    # list_rouge1[f'{task}_rouge1'] = metrics['rouge1']
                    # list_rouge2[f'{task}_rouge2'] = metrics['rouge2']
                    # list_rougeL[f'{task}_rougeL'] = metrics['rougeL']
                    # list_rougeLsum[f'{task}_rougeLsum'] = metrics['rougeLsum']
                

                total_loss = sum(list_loss) / len(list_loss)
                avg_f1  = sum([v for k, v in list_f1.items()]) / len(list_f1)
                result_metric = {**list_f1, 'avg_hete_f1': avg_f1}
            
            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return total_loss, result_metric

    return evaluate


def get_on_fit_config(save_path):
    """Return a function that will be used to construct the config that the client's
    fit() method will receive."""

    def fit_config_fn(server_round: int):
        fit_config = {}
        fit_config["current_round"] = server_round
        fit_config["save_path"] = save_path
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregate (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    total_flops = [m["flops"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": round(sum(losses) / sum(examples), 3), "total_flops": f"{sum(total_flops)/1e12:.2f}T"}


def server_fn(context: Context):
    """Construct components that set the ServerApp behaviour."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(os.getcwd(), f"results/{folder_name}")
    os.makedirs(save_path, exist_ok=True)

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    num_nodes = context.run_config['num-supernodes']
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))

    # Get initial model weights
    init_model, tokenizer = get_model(cfg.model)
    init_model_parameters = get_parameters(init_model)
    init_model_parameters = ndarrays_to_parameters(init_model_parameters)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=cfg.train.strategy.fraction_fit,
        fraction_evaluate=cfg.train.strategy.fraction_evaluate,
        on_fit_config_fn=get_on_fit_config(save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        initial_parameters=init_model_parameters,
        evaluate_fn=get_evaluate_fn(
            cfg.train, cfg.model, cfg.dataset, cfg.train.save_every_round, num_rounds, num_nodes, save_path
        ),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Flower ServerApp
app = ServerApp(server_fn=server_fn)