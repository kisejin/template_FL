from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
from transformers import BertForSequenceClassification, GenerationConfig, AutoTokenizer
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
            outputs.logits = self.dropout(outputs.logits)
        return outputs

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
        self, model, tokenizer, train_dataset, val_dataset, holdout_dataset, reference_dataset,
        args, data_collator, compute_metrics, mates_args, selection_fraction, data_influence_model, 
        data_influence_tokenizer
    ):
        self.accelerator = Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.mates_args = mates_args
        self.selection_fraction = selection_fraction
        self.data_influence_model = data_influence_model
        self.data_influence_tokenizer = data_influence_tokenizer

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

        if self.mates_args.state:
            self.influence_optimizer = torch.optim.AdamW(self.data_influence_model.parameters(), lr=self.args.learning_rate)
            # Prepare holdout and reference loaders for Accelerator
            self.data_influence_model, self.holdout_loader, self.reference_loader, self.influence_optimizer = self.accelerator.prepare(
                self.data_influence_model, self.holdout_loader, self.reference_loader, self.influence_optimizer
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

        for epoch in tqdm(range(self.args.num_train_epochs), 
                          bar_format='{l_bar}{bar} {percentage:3.0f}% |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
            self.model.train()
            epoch_loss = 0.0

            for step, batch in tqdm(enumerate(self.train_loader),
                                    bar_format='{l_bar}{bar} {percentage:3.0f}% |{n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                if step >= self.args.max_steps:
                    break

                # Check if it's time to update the data influence model and state is True
                if self.mates_args.state:
                    if step % self.mates_args.update_data_influence_model_step == 0:
                        print("Updating the data influence model and selecting high-quality data...")
                        self.update_data_influence_model()

                    print(f"Selection fraction: {self.selection_fraction}")

                    if self.selection_fraction < 1:
                        # Filter high-quality data using the data influence model
                        high_quality_indices = self.select_high_quality_data(
                            batch=batch,
                            selection_fraction=self.selection_fraction,
                        )
                        batch = {k: v[high_quality_indices] for k, v in batch.items()}

                self.optimizer.zero_grad()

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
            
        return {"training_loss": sum(training_loss) / len(training_loss), "best_val_loss": best_val_loss}
    

    def select_high_quality_data(self, batch, selection_fraction):
        """
        Use the data influence model to predict quality scores and select high-quality data indices.
        """
        print("Selecting high-quality data using the data influence model...")

        # Predict influence scores for the batch
        influence_scores = []
        self.data_influence_model.eval()

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
            outputs = self.data_influence_model(
                input_ids=bert_inputs['input_ids'],
                attention_mask=bert_inputs['attention_mask'],
            )
                
            influence_scores.extend(outputs.logits.squeeze(-1).cpu().numpy())

        end_time = time.perf_counter()
        runtime = round((end_time - start_time), 2)
        
        print(f'Time influence score prediction using SkipBERT: {runtime}')

        # Normalize influence scores and apply Gumbel-Top-$k$ selection
        influence_scores = np.array(influence_scores)
        print(">> Influence scores shape:", influence_scores.shape)

        # Add Gumbel noise for diversity
        rng = np.random.default_rng()
        gumbel_noise = rng.gumbel(size=len(influence_scores))
        influence_scores += gumbel_noise

        # Select top indices based on influence scores
        print(f"Selection fraction: {selection_fraction}")
        selection_size = int(len(influence_scores) * selection_fraction)
        selection_size = max(1, selection_size)  # Ensure at least one sample is selected
        print(f"List influence score: {influence_scores}, length: {len(influence_scores)}")
        print(f"Selection size: {selection_size}")
        selection_size = selection_size if len(influence_scores) != selection_size else selection_size - 1
        high_quality_indices = np.argpartition(influence_scores, selection_size)[:selection_size]
        print(f"Selected {len(high_quality_indices)} high-quality samples.")

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

        print("Starting to collect holdout-reference pairs...")
        self.model.train()

        for step, holdout_batch in enumerate(self.holdout_loader):
            print(f"Processing holdout batch {step+1}/{len(self.holdout_loader)}...")

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
            self.optimizer.step()

            # Use the trained (updated) model to compute reference losses
            print(f"Evaluating reference losses at step {step}...")
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
        self.model.load_state_dict(original_state, strict=False)

        # Train the data influence model using the generated pairs
        print("Starting to train the data influence model...")
        self.data_influence_model.train()

        for step, (text, score) in enumerate(holdout_reference_pairs):
            # Tokenize the text using the BERT tokenizer
            bert_inputs = self.data_influence_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=256,
                return_tensors='pt'
            ).to(self.accelerator.device)

            # Convert score to tensor and enable gradients
            score_tensor = torch.tensor([score], device=self.accelerator.device, dtype=torch.float32, requires_grad=True)
            
            # Train the data influence model
            self.influence_optimizer.zero_grad()
            outputs = self.data_influence_model(
                input_ids=bert_inputs['input_ids'],
                attention_mask=bert_inputs['attention_mask'],
                labels=score_tensor
            )
            influence_loss = outputs.loss

            self.accelerator.backward(influence_loss)

            if step % 50 == 0:
                print(f"[Influence Training] Step {step}: Loss = {influence_loss.item():.4f}")


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

