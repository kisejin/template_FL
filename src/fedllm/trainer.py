from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
from transformers import (
    BertForSequenceClassification, 
    GenerationConfig, 
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoModelForCausalLM,
    get_scheduler
)
from peft import prepare_model_for_kbit_training
import inspect
import logging
import wandb
from tqdm import tqdm
import time
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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

        for example in tqdm(sampled_dataset, desc="Generating Samples"):
            instruction = example.get("instruction", "")
            input_text = example.get("input", "")
            label = example.get("output", "")

            if input_text:
                prompt = f"Instruction: {instruction} Input: {input_text} Response:"
            else:
                prompt = f"Instruction: {instruction} Response:"

            prediction = self.generate(prompt)
            table.add_data(prompt, prediction, label, self.task)
        
        return table

    def log_samples_to_wandb(self, dataset):
        samples_table = self.create_samples_table(dataset)
        wandb.log({"sample_predictions": samples_table})


class ManualTrainer:
    def __init__(
        self, model, tokenizer, train_dataset, val_dataset, reference_dataset,
        args, data_collator, compute_metrics, mates_args, selection_fraction,
        **kwargs
    ):
        self.accelerator = Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.mates_args = mates_args
        self.selection_fraction = selection_fraction
        
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
            self.reference_dataset = self._remove_unused_columns(reference_dataset, "reference")

            self.reference_loader = DataLoader(
                self.reference_dataset,
                batch_size=self.mates_args.reference_batch_size,
                shuffle=False,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )

            # Prepare holdout and reference loaders for Accelerator
            self.reference_loader = self.accelerator.prepare(
                self.reference_loader
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

        num_training_steps = self.args.num_train_epochs * len(self.train_loader)
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=self.optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )

        for epoch in tqdm(
            range(self.args.num_train_epochs),
            total=self.args.num_train_epochs,
            desc="Epoch",
            bar_format='{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            colour="YELLOW"):
            self.model.train()
            epoch_loss = 0.0

            # Precompute update steps if state is active.
            # We want exactly num_data_influence_model_update updates during the epoch.
            # To do so, we first compute evenly spaced indices from 0 to (len(train_loader)-1)
            # with 2 extra points (to include endpoints), and then remove the endpoints.
            if self.mates_args.state:
                # full_update_indices = self._get_evenly_spaced_ints(len(self.train_loader) - 1,
                #                                             self.mates_args.num_data_influence_model_update + 2)
                # # Exclude the first (0) and last (len(train_loader)-1) indices:
                # update_steps = set(full_update_indices[1:-1])

                self.prune_dataset_by_perplexity(selection_criteria=self.mates_args.selection_criteria)
            
            for step, batch in tqdm(enumerate(self.train_loader),
                                    total=len(self.train_loader),
                                    desc="Training",
                                    bar_format='{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                                    colour="BLUE",
                                    unit=" samples"):
                if step >= self.args.max_steps and self.args.max_steps > 0:
                    break

                

                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                loss = outputs.loss

                self.accelerator.backward(loss)
                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()

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
    

    def train_reference_model(self):
        """
        Trains the reference model on the reference dataset.
        Applies quantization before training for better efficiency.
        """
        if not self.mates_args.state or not hasattr(self, "reference_loader"):
            print("Reference loader not available. Skipping reference model training.")
            return

        print("Training reference model...")
        
        # First create a copy of the model
        model_to_quantize = copy.deepcopy(self.model)
        
        # Configure quantization based on specified bit precision
        quantization_bit = getattr(self.mates_args, "quantization_bit", 8)  # Default to 8-bit for training
        
        ref_epochs = self.args.num_train_epochs
        
        # For training, we should use 8-bit as 4-bit typically doesn't support training
        if quantization_bit != -1:
            if quantization_bit != 8 and self.accelerator.is_main_process:
                print(f"Warning: {quantization_bit}-bit quantization may not support training.")
                print("Switching to 8-bit quantization which better supports training.")
                quantization_bit = 8
            
            print(f"Applying {quantization_bit}-bit quantization before training...")
            
            try:
                from transformers import BitsAndBytesConfig
                
                # Create quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4",
                    bnb_8bit_enable_fp32_cpu_offload=True  # Enable FP32 offload for stability during training
                )
                
                # Get model config
                model_config = model_to_quantize.config
                
                # Save model path if available
                model_path = getattr(model_to_quantize, "name_or_path", None)
                
                # If no path available, save to temp directory
                if not model_path:
                    import os, tempfile
                    temp_dir = tempfile.mkdtemp()
                    model_to_quantize.save_pretrained(temp_dir)
                    model_path = temp_dir
                
                # Reload with quantization
                self.reference_model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    config=model_config,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                print(f"Successfully quantized reference model to {quantization_bit}-bit")
                
            except Exception as e:
                print(f"Quantization failed: {e}")
                print("Falling back to full precision model")
                self.reference_model = model_to_quantize
            
            # Prepare optimizer for reference model
            # Use optimizer with 8-bit Adam which is compatible with quantized models
            try:
                import bitsandbytes as bnb
                ref_optimizer = bnb.optim.AdamW8bit(
                    self.reference_model.parameters(),
                    lr=self.args.learning_rate
                )
                print("Using 8-bit optimizer for quantized model")
            except:
                ref_optimizer = torch.optim.AdamW(
                    self.reference_model.parameters(),
                    lr=self.args.learning_rate
                )
                print("Using standard optimizer")
        
        else:
            
            from bitsandbytes.optim import Lion
            
            # Get model config
            model_config = model_to_quantize.config
            
            # Save model path if available
            model_path = getattr(model_to_quantize, "name_or_path", None)
            
            # If no path available, save to temp directory
            if not model_path:
                import os, tempfile
                temp_dir = tempfile.mkdtemp()
                model_to_quantize.save_pretrained(temp_dir)
                model_path = temp_dir
            
            # Reload with quantization
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                config=model_config,
                device_map="auto"
            )

            print(f"Successfully loaded reference model")
            
            ref_optimizer = Lion(
                self.reference_model.parameters(),
                lr=self.mates_args.learning_rate,
                weight_decay=self.mates_args.weight_decay
            )
    
        gradient_accumulation_steps = 1
        num_update_steps_per_epoch = len(self.reference_loader) // gradient_accumulation_steps
        num_training_steps = ref_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            name="cosine",
            optimizer=ref_optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps,
        )
        
        # Prepare reference model and optimizer with accelerator
        if self.mates_args.quantization_bit in [4, 8]:
            self.reference_model = prepare_model_for_kbit_training(self.reference_model)
        self.reference_model, ref_optimizer, lr_scheduler, self.reference_loader, self.train_loader = self.accelerator.prepare(
            self.reference_model, ref_optimizer, lr_scheduler, self.reference_loader, self.train_loader
        )
        
        # Train the quantized model
        
        print("Training reference model...")
        
        # Start time
        start_time = time.time()
        self.reference_model.train()
        
        for epoch in tqdm(
            range(ref_epochs), 
            total=ref_epochs,
            desc="Reference EPOCH",
            bar_format='{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
            colour="YELLOW"
        ):
            total_loss = 0.0
            
            pbar = tqdm(
                self.reference_loader,
                total=len(self.reference_loader),
                desc="Reference BATCH",
                bar_format='{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}/{postfix}]',
                colour="GREEN",
                unit=" samples",
            )
            
            for batch in pbar:
                
                batch = {k: v.to("cuda") for k, v in batch.items()}
                
                outputs = self.reference_model(**batch)
                loss = outputs.loss
                
                # Use accelerator for backward pass
                self.accelerator.backward(loss)
                ref_optimizer.step()
                lr_scheduler.step()
                ref_optimizer.zero_grad()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.reference_loader)
            print(f"Reference model - Epoch {epoch+1}/{ref_epochs}, Loss: {avg_loss:.4f}")
        
        # End time
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Reference model training completed in {runtime:.2f} seconds")
        
        self.reference_model.eval()



    def prune_dataset_by_perplexity(self, selection_criteria="high"):
        """
        Prunes the training dataset based on perplexity scores from reference model.
        """
        self.train_reference_model()
        self.reference_model.eval()
        perplexity_dict = {}
        
        print(f"Computing perplexity for training samples using reference model...")
        
        # Compute perplexity for each sample with optimized batch processing
        batch_indices = []
        batch_perplexities = []
        
        # Start time
        start_time = time.time()
        
        with torch.no_grad():
            pbar = tqdm(
                self.train_loader,
                total=len(self.train_loader),
                desc="Computing Perplexity",
                bar_format='{l_bar}{bar} |{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}/{postfix}]',
                colour="BLUE",
                unit=" samples",
            )
            for idx, batch in enumerate(pbar):
                batch = {k: v.to("cuda") for k, v in batch.items()}
                

                outputs = self.reference_model(**batch)
                
                logits = outputs.logits
                labels = batch['labels']
                bs, seq_len, vocab_size = logits.shape
                
                loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100, reduction='none')
                
                flat_loss = loss.reshape(bs, -1)
                
                # Create mask for padding
                mask = (labels != -100).float()
                
                # Apply mask to loss
                loss_per_sample = (flat_loss * mask).sum(dim=1) / seq_len
                
                perplexity = torch.exp2(loss_per_sample)
                
                batch_perplexities.extend(perplexity.tolist())
                
                # perplexity_dict[idx] = perplexity
                
                if idx % 100 == 0:
                    print(f"Processed {idx}/{len(self.train_loader)} batches")
        
        # Sort samples by perplexity score
        perplexity_dict = {idx: perplexity for idx, perplexity in enumerate(batch_perplexities)}
        sorted_indices = sorted(perplexity_dict.keys(), key=lambda x: perplexity_dict[x])
        
        # Select indices based on criteria
        total_samples = len(sorted_indices)
        num_samples_to_keep = int(total_samples * self.selection_fraction)
        
        if selection_criteria == "low":
            selected_indices = sorted_indices[:num_samples_to_keep]
        elif selection_criteria == "medium":
            mid_point = total_samples // 2
            start_idx = mid_point - (num_samples_to_keep // 2)
            end_idx = start_idx + num_samples_to_keep
            selected_indices = sorted_indices[start_idx:end_idx]
        elif selection_criteria == "high":
            selected_indices = sorted_indices[-(total_samples - num_samples_to_keep):]
        else:
            raise ValueError(f"Unknown selection criteria: {selection_criteria}")
        
        # Create pruned dataset
        pruned_dataset = self.train_dataset.select(selected_indices)
        
        # Create new train loader with pruned dataset
        new_train_loader = DataLoader(
            pruned_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last
        )
        
        # Prepare the new loader with accelerator
        self.train_loader = self.accelerator.prepare(new_train_loader)
        self.train_dataset = pruned_dataset

        torch.cuda.empty_cache()
        
        # End time
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Dataset pruning completed in {runtime:.2f} seconds")
        
        print(f"Dataset pruned: {len(pruned_dataset)}/{total_samples} samples kept ({selection_criteria} selection)")
        
    def evaluate(self, wandb_sample=False):
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

                logits = logits.cpu().numpy()
                labels = labels.cpu().numpy()

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
        print("Validation Metrics:", metrics)

        if wandb_sample:
            # Sample Logging
            llm_sample_cb = ManualLLMSampleCB(
                model=self.model,
                tokenizer=self.tokenizer,
                task="classification",
                num_samples=5,
                max_new_tokens=128
            )
            llm_sample_cb.log_samples_to_wandb(self.val_dataset)

        return metrics