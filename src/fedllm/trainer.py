from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
import time


from transformers import (
    # BertForSequenceClassification, 
    GenerationConfig, 
    AutoTokenizer,
    Trainer,
    get_scheduler, 
    EarlyStoppingCallback,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from .skipbert.trainer import compute_metrics_skipbert, SkipBertTrainer

import inspect
import logging
import wandb
from tqdm import tqdm

from functools import partial
import torch.nn.functional as F

logger = logging.getLogger(__name__)

device_map = "cuda" if torch.cuda.is_available() else "cpu"

# Wrapper to add dropout to the model's outputs (e.g. logits)
class ModelWithDropoutWrapper(torch.nn.Module):
    def __init__(self, model, dropout_p):
        super().__init__()
        self.model = model
        self.dropout = torch.nn.Dropout(dropout_p)
    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        # If outputs has logits, apply dropout to them
        if hasattr(outputs, "logits") and outputs.logits is not None:
            outputs.logits = self.dropout(outputs.logits.to(self.model.dtype))
        return outputs
    
def time_format(runtime, logger):

    if runtime < 60:
        # logger.info(f'Runtime: {runtime:.2f} seconds')
        print(f'Runtime: {runtime:.2f} seconds')
    elif runtime < 3600:  # Less than one hour
        minutes = runtime / 60
        # logger.info(f'Runtime: {minutes:.2f} minutes')
        print(f'Runtime: {minutes:.2f} minutes')
    else:
        hours = runtime / 3600
        # logger.info(f'Runtime: {hours:.2f} hours')
        print(f'Runtime: {minutes:.2f} minutes')

        
def convert_to_tokens_reg(data, tokenizer, max_seq_length, device):
    input_tokenzied = tokenizer(data['text'], truncation=True, padding=True, max_length=max_seq_length, return_tensors="pt")
    input_tokenzied['labels'] = torch.tensor(data['label'], dtype=torch.float32).reshape(-1, 1)

    return input_tokenzied

class ManualLLMSampleCB:
    def __init__(self, model, tokenizer, task, num_samples=10, max_new_tokens=256):
        self.model = model
        self.concat_model = None
        self.tokenizer = tokenizer
        self.task = task
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.gen_config = GenerationConfig.from_pretrained(
            model.config.name_or_path, max_new_tokens=max_new_tokens
        )

    def generate(self, prompt):
        # Tokenize the input prompt and include the attention mask
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_ids = tokenized_prompt['input_ids']
        attention_mask = tokenized_prompt['attention_mask']  # Extract attention mask

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_new_tokens,
                generation_config=self.gen_config
            )
        return self.tokenizer.decode(output[0], skip_special_tokens=True)


    def create_samples_table(self, dataset):
        table = wandb.Table(columns=["input", "prediction", "label", "task"])
        sampled_dataset = dataset.shuffle(seed=42).select(range(self.num_samples))

        for example in tqdm(sampled_dataset, 
                            desc="Generating Samples",
                           bar_format='{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            label = example.get("output", "")

            if input_text:
                prompt = f"{instruction} \n {input_text}"
            else:
                prompt = f"{instruction}"

                
            prompt = prompt.split("### Response:")[0] + "\n### Response:"
            prediction = self.generate(prompt)
            prediction = prediction.split("### Response:")[-1]
            label = label.split("### Response:")[-1]
            
            table.add_data(prompt, prediction, label, self.task)
        
        return table

    def log_samples_to_wandb(self, dataset):
        samples_table = self.create_samples_table(dataset)
        wandb.log({"sample_predictions": samples_table})


class ManualTrainer:
    def __init__(
        self, model, tokenizer, 
        train_dataset, val_dataset, holdout_dataset, reference_dataset,
        args, data_collator, compute_metrics, mates_args, skipbert_args, 
        selection_fraction, 
        teacher_data_influence_model, student_data_influence_model, 
        data_influence_tokenizer,task
    ):

        self.accelerator = Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.mates_args = mates_args
        self.skipbert_args = skipbert_args
        self.selection_fraction = selection_fraction
        self.teacher_data_influence_model = teacher_data_influence_model
        self.student_data_influence_model = student_data_influence_model
        self.data_influence_tokenizer = data_influence_tokenizer
        self.task = task

        # Remove unused columns from datasets
        if train_dataset:
            self.train_dataset = self._remove_unused_columns(train_dataset, "training")     
            # Prepare data loaders
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )

        else:
            self.train_loader = None

            
        if val_dataset:
            self.val_dataset = self._remove_unused_columns(val_dataset, "validation")          
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )
        else:
            self.val_loader = None
        
        
        if self.mates_args.state:
            self.holdout_dataset = self._remove_unused_columns(holdout_dataset, "holdout")
            self.reference_dataset = self._remove_unused_columns(reference_dataset, "reference")

            self.holdout_loader = DataLoader(
                self.holdout_dataset,
                batch_size=self.mates_args.holdout_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )

            self.reference_loader = DataLoader(
                self.reference_dataset,
                batch_size=self.mates_args.reference_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )

        # Prepare optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate
        )

        # Prepare model, optimizer, and data loaders for Accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader
        )
        
        ### Define for MATEs ###
        if self.mates_args.state:
            
            # Prepare holdout and reference loaders for Accelerator
            self.teacher_influence_optimizer = torch.optim.AdamW(self.teacher_data_influence_model.parameters(), lr=self.args.learning_rate)
            
            self.data_influence_model, self.holdout_loader, self.reference_loader, self.teacher_influence_optimizer = self.accelerator.prepare(
                self.teacher_data_influence_model, self.holdout_loader, self.reference_loader, self.teacher_influence_optimizer
            )
            
            ######
        
        ### Define for SkipBERT ###
        # Create a config with use_configured_state set to True
        
        self.skipbert_train_args = TrainingArguments(
            output_dir=self.skipbert_args.output_dir,
            learning_rate=self.skipbert_args.learning_rate,
            num_train_epochs=self.skipbert_args.num_train_epochs,
            per_device_train_batch_size=self.skipbert_args.train_batch_size,
            gradient_accumulation_steps=self.skipbert_args.gradient_accumulation_steps,
            per_device_eval_batch_size=self.skipbert_args.eval_batch_size,
            eval_accumulation_steps=self.skipbert_args.eval_accumulation_steps,
            max_steps=self.skipbert_args.max_steps,
            logging_steps = 10,
            evaluation_strategy=self.skipbert_args.evaluation_strategy,
            save_strategy=self.skipbert_args.save_strategy,
            lr_scheduler_type=self.skipbert_args.lr_scheduler_type,
            warmup_steps=self.skipbert_args.warmup_steps,
            weight_decay=self.skipbert_args.weight_decay,
            logging_dir=self.skipbert_args.logging_dir,
            report_to='wandb',
            run_name='skipbert',
            do_train=self.skipbert_args.do_train,
            do_eval=self.skipbert_args.do_eval,
            dataloader_drop_last=False,
            ddp_find_unused_parameters=False,
            group_by_length=True,
            load_best_model_at_end = True,
            accelerator_config={"use_configured_state": True}
        )

        

        # Prepare custom optimizer student model's parameters
        if self.student_data_influence_model is not None:
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            self.student_optimizer_grouped_parameters = [
                {
                    'params': [p for n, p in self.student_data_influence_model.named_parameters() if not any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.01
                },
                {
                    'params': [p for n, p in self.student_data_influence_model.named_parameters() if any(nd in n for nd in no_decay)], 
                    'weight_decay': 0.0
                }
            ]

        ######

    def _remove_unused_columns(self, dataset, description=None):
        """
        Removes columns from a dataset that are not used by the model's forward method.
        
        Args:
            dataset: A dataset object (e.g., from datasets.Dataset).
            description: A string description of the dataset (e.g., "training" or "validation").
        Returns:
            The dataset with unused columns removed.
        """
        # Inspect the model forward signature
        forward_signature = inspect.signature(self.model.forward)
        signature_columns = list(forward_signature.parameters.keys())

        # Add label columns to the signature columns
        label_columns = ["labels", "label_ids"]
        signature_columns += label_columns

        # Determine unused columns
        dataset_columns = set(dataset.column_names)
        used_columns = set(signature_columns).intersection(dataset_columns)
        ignored_columns = list(dataset_columns - used_columns)

        if ignored_columns:
            logger.info(
                f"The following columns in the {description} set don't have a corresponding argument in "
                f"`{self.model.__class__.__name__}.forward` and have been ignored: {', '.join(ignored_columns)}."
            )

        # Ensure at least one column matches the model's expected inputs
        if not used_columns:
            raise ValueError(
                f"No columns in the {description} dataset match the model's forward method signature. "
                f"The following columns have been ignored: {', '.join(ignored_columns)}."
            )

        return dataset.remove_columns(ignored_columns)
    
    def _get_evenly_spaced_ints(self, n, num_updates):
        """
        Returns num_updates integers approximately evenly spaced between 0 and n (inclusive).
        If the exact spacing is not an integer, the intermediate values are rounded to the nearest integer.
        """
        if num_updates == 1:
            return [0]
        
        step = n / (num_updates - 1)
        result = []
        for i in range(num_updates):
            # Always force the first and last elements to be exactly 0 and n.
            if i == 0:
                result.append(0)
            elif i == num_updates - 1:
                result.append(n)
            else:
                result.append(round(i * step))
        return result

    def train(self):
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 5
        training_loss = []
        metric_scores = {
            'f1': [],
            'rouge1': [],
            'rouge2': [],
            'rougeL': [],
            'rougeLsum': [],
        }
        
        print(f"Selection fraction: {self.selection_fraction}")
        
        for epoch in tqdm(range(self.args.num_train_epochs),
                            total=self.args.num_train_epochs,
                            desc="Epoch",
                            bar_format='{l_bar}{bar} {percentage:3.0f}% |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            self.model.train()
            epoch_loss = 0.0

            # Precompute update steps if state is active.
            # We want exactly num_data_influence_model_update updates during the epoch.
            # To do so, we first compute evenly spaced indices from 0 to (len(train_loader)-1)
            # with 2 extra points (to include endpoints), and then remove the endpoints.
            if self.mates_args.state:
                full_update_indices = self._get_evenly_spaced_ints(len(self.train_loader) - 1,
                                                            self.mates_args.num_data_influence_model_update + 2)
                # Exclude the first (0) and last (len(train_loader)-1) indices:
                update_steps = set(full_update_indices[1:-1])
            
            for step, batch in tqdm(enumerate(self.train_loader),
                                    total=len(self.train_loader),
                                    desc="Step",
                                    bar_format='{l_bar}{bar} {percentage:3.0f}% |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                
                if step >= self.args.max_steps and self.args.max_steps > 0:
                    break

                # If state is active and the current step is one of the precomputed update steps,
                # update the data influence model.
                if self.mates_args.state and step in update_steps:
                    print("Updating the data influence model and selecting high-quality data...")
                    self.update_data_influence_model()

                    if self.selection_fraction < 1.0:
                        # Filter high-quality data using the data influence model
                        high_quality_indices = self.select_high_quality_data(
                            batch=batch,
                            selection_fraction=self.selection_fraction,
                        )
                        batch = {k: v[high_quality_indices] for k, v in batch.items()}

                self.optimizer.zero_grad()
                torch.cuda.empty_cache()

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss

                self.accelerator.backward(loss)
                self.optimizer.step()

                epoch_loss += loss.item()

                if (step + 1) % self.args.logging_steps == 0:
                    print(f"Step {step + 1}: Train Loss = {epoch_loss / (step + 1):.4f}")

            avg_epoch_loss = epoch_loss / len(self.train_loader)
            training_loss.append(avg_epoch_loss)

            val_results = self.evaluate()
            for name, score in val_results.items():
                if name != 'eval_loss':
                    metric_scores[name].append(score)

            print(f"Epoch {epoch + 1}: Train Loss = {avg_epoch_loss:.4f}, Val Loss = {val_results['eval_loss']:.4f}")

            # Early stopping logic
            if val_results["eval_loss"] < best_val_loss:
                best_val_loss = val_results["eval_loss"]
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
        index = metric_scores['f1'].index(max(metric_scores['f1']))
        return {
            "training_loss": sum(training_loss) / len(training_loss), 
            "eval_loss": best_val_loss,
            "eval_scores": {k: v[index] for k, v in metric_scores.items()}
        }

    def select_high_quality_data(self, batch, selection_fraction):
        """
        Use the data influence model to predict quality scores and select high-quality data indices.
        """
        # print("Selecting high-quality data using the data influence model...")

        # Predict influence scores for the batch
        influence_scores = []
        self.student_data_influence_model.eval()

        start_time = time.perf_counter()
        
        with torch.no_grad():
            text = self.tokenizer.batch_decode(
                batch['input_ids'], 
                skip_special_tokens=True
            )
            
            # Tokenize the text using the BERT tokenizer
            bert_inputs = self.data_influence_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            ).to(self.accelerator.device)
            
            # Get influence scores from the data influence model
            logits, attn_outputs, hidn_output = self.student_data_influence_model(
                input_ids=bert_inputs['input_ids'],
                attention_mask=bert_inputs['attention_mask'],
            )
            
            influence_scores.extend(logits.squeeze(-1).cpu().numpy())

        end_time = time.perf_counter()
        runtime = round((end_time - start_time), 2)
        
        # print('Time influence score prediction using SkipBERT: ')
        time_format(runtime, logger)

        # Normalize influence scores and apply Gumbel-Top-$k$ selection
        influence_scores = np.array(influence_scores)
        # print(">> Influence scores shape:", influence_scores.shape)

        # Add Gumbel noise for diversity
        rng = np.random.default_rng()
        gumbel_noise = rng.gumbel(size=len(influence_scores))
        influence_scores += gumbel_noise

        # Select top indices based on influence scores
        # print(f"Selection fraction: {selection_fraction}")
        
        selection_size = int(len(influence_scores) * selection_fraction)
        selection_size = max(1, selection_size)  # Ensure at least one sample is selected
        
        # print(f"Length influence score: {len(influence_scores)}")
        # print(f"Selection size: {selection_size}")
        selection_size = selection_size if len(influence_scores) != selection_size else selection_size - 1
        high_quality_indices = np.argpartition(influence_scores, selection_size)[:selection_size]
        # print(f"Selected {len(high_quality_indices)} high-quality samples.")

        return high_quality_indices

    def create_filtered_dataloader(self, indices):
        """
        Create a new dataloader with only the selected high-quality data.
        """
        print("Creating a filtered dataloader with selected high-quality data...")
        subset_dataset = torch.utils.data.Subset(self.train_dataset, indices)
        return torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,  # Use the same collate function
            drop_last=self.args.dataloader_drop_last
        )


    def update_data_influence_model(self):
        # Save the original (untrained) state of self.model.
        original_state = copy.deepcopy(self.model.state_dict())
        holdout_reference_pairs = []

        torch.cuda.empty_cache()

        # Wrap the model with dropout before training on holdout data.
        self.model = ModelWithDropoutWrapper(self.model, dropout_p=self.mates_args.copied_model_dropout_rate)

        # print("Starting to collect holdout-reference pairs...")
        self.model.train()
        

        for step, holdout_batch in tqdm(enumerate(self.holdout_loader),
                                        total=len(self.holdout_loader),
                                        bar_format='{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            # print(f"Processing holdout batch {step+1}/{len(self.holdout_loader)}...")

            self.optimizer.zero_grad()
            # Train on the holdout batch (this updates the wrapped model temporarily)
            outputs = self.model(
                input_ids=holdout_batch['input_ids'],
                attention_mask=holdout_batch['attention_mask'],
                labels=holdout_batch['labels']
            )
            holdout_loss = outputs.loss
            decoded_texts = self.tokenizer.batch_decode(
                holdout_batch['input_ids'], 
                skip_special_tokens=True
            )

            self.accelerator.backward(holdout_loss)
            # holdout_loss.backward()
            self.optimizer.step()

            # Use the trained (updated) model to compute reference losses
            # print(f"Evaluating reference losses at step {step}...")
            self.model.eval()
            reference_losses = []

            with torch.no_grad():
                for ref_batch in self.reference_loader:
                    outputs = self.model(
                        input_ids=ref_batch['input_ids'],
                        attention_mask=ref_batch['attention_mask'],
                        labels=ref_batch['labels']
                    )
                    reference_losses.append(outputs.loss.item())

            # Compute the mean of reference losses
            score = sum(reference_losses) / len(reference_losses) if reference_losses else 0.0
            holdout_reference_pairs.append((decoded_texts, score))
            self.model.train()

        # Restore self.model to its original (untrained) state.
        self.model = self.model.model
        original_state = {k: v.to(self.model.dtype) for k, v in original_state.items()}
        self.model.load_state_dict(original_state, strict=False)

        # Train the data influence model using the generated pairs
        print("Starting to train the data influence model...")
        self.teacher_data_influence_model.train()
        
        # Convert to HF datasets
        list_texts, list_score = [], []
        batch_size = 0

        # Convert to Dataset objective
        for texts, score in holdout_reference_pairs:
            if batch_size == 0:
                batch_size = len(texts)
            list_texts.extend(texts)
            list_score.extend([score] * len(texts))

        holdout_reference_pairs = {'text': list_texts, 'label': list_score}
        holdout_reference_pairs = Dataset.from_dict(holdout_reference_pairs)

        

        # Wrap the function with partial
        convert_func = partial(
            convert_to_tokens_reg,
            tokenizer=self.data_influence_tokenizer,
            max_seq_length=self.skipbert_args.max_seq_length,
            device=self.accelerator.device
        )

        holdout_reference_pairs_loader = DataLoader(
            holdout_reference_pairs.map(
                convert_func,
                batched=True,
                num_proc=8,
                remove_columns=holdout_reference_pairs.column_names
            ), 
            batch_size=batch_size,
            collate_fn=DataCollatorWithPadding(tokenizer=self.data_influence_tokenizer, padding=True, max_length=self.skipbert_args.max_seq_length),  # Use the same collate function
            drop_last=self.args.dataloader_drop_last
        )
        
        
        
        for epoch in range(self.mates_args.data_influence_model_epochs):
            print(f"Epoch {epoch + 1}/{self.mates_args.data_influence_model_epochs}")
            
            for step, batch_input in tqdm(
                                        enumerate(holdout_reference_pairs_loader),
                                        bar_format='{l_bar}{bar} {percentage:3.0f}% | {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'
                                    ):      # Tokenize the text using the BERT tokenizer

                batch_input = {k: v.to('cuda') for k, v in batch_input.items()} # cuda:0
                # bert_inputs = self.data_influence_tokenizer(
                #     text,
                #     truncation=True,
                #     padding='max_length',
                #     max_length=256,
                #     return_tensors='pt'
                # ).to(self.accelerator.device)


                # Convert score to tensor and enable gradients
                # score_tensor = torch.tensor([score], device=self.accelerator.device, dtype=torch.float32, requires_grad=True)

                # Train the data influence model
                self.teacher_influence_optimizer.zero_grad()
                outputs = self.teacher_data_influence_model(
                    # **batch_input
                    input_ids=batch_input['input_ids'],
                    attention_mask=batch_input['attention_mask'],
                    labels=batch_input['labels'].view(-1)
                )

                # influence_loss = loss_mse(batch_input['labels'].view(-1), outputs.logits.view(-1))
                # print(f"Loss: {influence_loss} - require_grad: {influence_loss.grad_fn}")

                influence_loss = outputs.loss

                self.accelerator.backward(influence_loss)
                self.teacher_influence_optimizer.step()

                if step % 50 == 0:
                    print(f"[Influence Training] Step {step}: Loss = {influence_loss.item():.4f}")
            
        
        ### Distillation for SkipBERT ###
        train_converted = holdout_reference_pairs.map(
            convert_func,
            batched=True,
            num_proc=8,
            remove_columns=holdout_reference_pairs.column_names
        )

        
        # Call parent constructor with custom optimizer
        optimizer = torch.optim.AdamW(
            self.student_optimizer_grouped_parameters, 
            lr=self.skipbert_train_args.learning_rate,
        )

        scheduler = get_scheduler(
            name=self.skipbert_train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=self.skipbert_train_args.warmup_steps,
            # num_training_steps=training_args.max_steps
            num_training_steps=100/(self.skipbert_train_args.per_device_train_batch_size * self.skipbert_train_args.gradient_accumulation_steps)

        )


        # Initialize the trainer
        trainer = SkipBertTrainer(
            student_model=self.student_data_influence_model,
            teacher_model=self.teacher_data_influence_model,
            args=self.skipbert_train_args,
            train_dataset=train_converted,
            eval_dataset=train_converted.shuffle().select(range(min(len(train_converted),10))),

            compute_metrics=compute_metrics_skipbert,
            # SkipBERT specific arguments
            alpha=0.5,
            temperature=2.0,
            beta=1.0,
            use_logits=self.skipbert_args.use_logits,
            use_att=self.skipbert_args.use_att,
            use_rep=self.skipbert_args.use_rep,
            use_embedding=self.skipbert_args.use_embedding,
            att_layer_maps=self.skipbert_args.att_layer_maps,
            hid_layer_maps=self.skipbert_args.hid_layer_maps,
            epochs_no_cls=self.skipbert_args.epochs_no_cls,
            reduce_T=self.skipbert_args.reduce_T,
            output_mode=self.skipbert_args.output_mode, # 'classification' or 'regression'
            num_masked_layers_teacher=self.skipbert_args.num_masked_layers_teacher,
            num_masked_last_layers_teacher=self.skipbert_args.num_masked_last_layers_teacher,
            fp16=self.skipbert_args.fp16,
            num_full_hidden_layers_student=self.skipbert_args.num_full_hidden_layers_student,
            tokenizer=self.data_influence_tokenizer,
            optimizers=(optimizer,scheduler),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]

        )

        # Train the model
        print(f"### KD student data influence model ###")
        start_time = time.perf_counter()
        trainer.train()
        end_time = time.perf_counter()
        runtime = round((end_time - start_time), 2)


    def evaluate(self, wandb_sample=True):
        self.model.eval()
        val_loss = 0.0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                val_loss += outputs.loss.item()

                logits = self.accelerator.gather(outputs.logits)
                labels = self.accelerator.gather(batch['labels'])

                logits = logits.to(torch.float32).cpu().numpy()

                labels = labels.to(torch.int32).cpu().numpy()
                predictions = np.argmax(logits, axis=-1)
                attention_mask = batch['attention_mask'].cpu().numpy()

                for pred, label, mask in zip(predictions, labels, attention_mask):
                    valid_pred = pred[mask.astype(bool)]
                    valid_label = label[mask.astype(bool)]

                    all_preds.append(valid_pred)
                    all_labels.append(valid_label)

        max_len = max(len(seq) for seq in all_preds)
        padded_preds = np.array([
            np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=self.tokenizer.pad_token_id)
            for seq in all_preds
        ])

        max_len = max(len(seq) for seq in all_labels)
        padded_labels = np.array([
            np.pad(seq, (0, max_len - len(seq)), 'constant', constant_values=-100)
            for seq in all_labels
        ])

        metrics = self.compute_metrics({"predictions": padded_preds, "label_ids": padded_labels})

        metrics.update({"eval_loss": val_loss / len(self.val_loader)})
        print(f"Validation Metrics: {metrics}")

        if wandb_sample:
            # Sample Logging
            instruction = self.tokenizer.batch_decode(self.val_dataset['input_ids'])
            output = self.tokenizer.batch_decode(self.val_dataset['labels'])
            valid_ds = {
                'instruction': instruction,
                'output': output
            }

            valid_ds = Dataset.from_dict(valid_ds)
            
            llm_sample_cb = ManualLLMSampleCB(
                model=self.model,
                tokenizer=self.tokenizer,
                task=self.task,
                num_samples=5,
                max_new_tokens=128
            )
            llm_sample_cb.log_samples_to_wandb(valid_ds)

        return metrics

