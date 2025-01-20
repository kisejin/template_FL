from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import copy
import numpy as np
from transformers import BertForSequenceClassification, GenerationConfig
import inspect
import logging
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)

class ManualLLMSampleCB:
    def __init__(self, model, tokenizer, task, num_samples=10, max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.task = task
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.gen_config = GenerationConfig.from_pretrained(
            model.config.name_or_path, max_new_tokens=max_new_tokens
        )

    def generate(self, prompt):
        tokenized_prompt = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        input_ids = tokenized_prompt['input_ids']

        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
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
        args, data_collator, compute_metrics, use_mates
    ):
        self.accelerator = Accelerator()
        self.model = model
        self.tokenizer = tokenizer
        self.args = args
        self.data_collator = data_collator
        self.compute_metrics = compute_metrics
        self.use_mates = use_mates

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

        if use_mates:
            self.holdout_dataset = self._remove_unused_columns(holdout_dataset, "holdout")
            self.reference_dataset = self._remove_unused_columns(reference_dataset, "reference")

        if use_mates:
            self.holdout_loader = DataLoader(
                self.holdout_dataset,
                batch_size=self.args.per_device_train_batch_size,
                shuffle=True,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last
            )

            self.reference_loader = DataLoader(
                self.reference_dataset,
                batch_size=self.args.per_device_eval_batch_size,
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

        if self.use_mates:
            # Prepare holdout and reference loaders for Accelerator
            self.holdout_loader, self.reference_loader = self.accelerator.prepare(
                self.holdout_loader, self.reference_loader
            )
            # Initialize data influence model
            self.data_influence_model = self.initialize_data_influence_model()

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

    def initialize_data_influence_model(self):
        # Initialize the data influence model for predicting data influence
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        return self.accelerator.prepare(model)

    def train(self):
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 5
        training_loss = []

        for epoch in range(self.args.num_train_epochs):
            # Check if it's time to update the data influence model and use_mates is True
            if self.use_mates and epoch % self.args.save_steps == 0:
                self.update_data_influence_model()

            self.model.train()
            epoch_loss = 0.0

            for step, batch in enumerate(self.train_loader):
                if step >= self.args.max_steps:
                    break

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

        return {"training_loss": sum(training_loss)/len(training_loss), "best_val_loss": best_val_loss}

    def update_data_influence_model(self):
        # Train a copy of the model on holdout data and validate on reference data
        copied_model = copy.deepcopy(self.model)
        copied_model.train()
        optimizer = torch.optim.Adam(copied_model.parameters(), lr=self.args.learning_rate)

        for holdout_batch in self.holdout_loader:
            optimizer.zero_grad()
            outputs = copied_model(
                input_ids=holdout_batch['input_ids'],
                attention_mask=holdout_batch['attention_mask'],
                labels=holdout_batch['labels']
            )
            holdout_loss = outputs.loss
            holdout_loss.backward()
            optimizer.step()

        copied_model.eval()
        reference_losses = []
        with torch.no_grad():
            for ref_batch in self.reference_loader:
                outputs = copied_model(
                    input_ids=ref_batch['input_ids'],
                    attention_mask=ref_batch['attention_mask'],
                    labels=ref_batch['labels']
                )
                reference_losses.append(outputs.loss.item())

        # Train the data influence model using reference data and their losses
        self.data_influence_model.train()
        influence_optimizer = torch.optim.AdamW(self.data_influence_model.parameters(), lr=self.args.learning_rate)
        for ref_batch, loss in zip(self.reference_loader, reference_losses):
            # Move tensors to CPU for printing
            input_ids = ref_batch['input_ids'].to("cpu")
            attention_mask = ref_batch['attention_mask'].to("cpu")

            # Print shapes and types of tensors
            print(f"input_ids shape: {input_ids.shape}")
            print(f"attention_mask shape: {attention_mask.shape}")
            print(f"labels shape: (1,), dtype: float")

            # Move tensors back to the appropriate device for training
            input_ids = input_ids.to(self.accelerator.device)
            attention_mask = attention_mask.to(self.accelerator.device)

            influence_optimizer.zero_grad()
            outputs = self.data_influence_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=torch.tensor([float(loss)], device=self.accelerator.device)
            )
            influence_loss = outputs.loss
            influence_loss.backward()
            influence_optimizer.step()


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

        metrics = self.compute_metrics({
            "predictions": padded_preds,
            "label_ids": padded_labels
        })

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

