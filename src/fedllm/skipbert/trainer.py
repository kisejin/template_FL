import copy
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import (
    AutocastKwargs,
    DistributedDataParallelKwargs,
    DistributedType,
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset
from transformers import (  # BertForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    get_scheduler,
)
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import EvaluationStrategy, IntervalStrategy
from transformers.training_args import OptimizerNames
from transformers.utils import is_sagemaker_mp_enabled

logging.getLogger("Trainer").setLevel(logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics_skipbert(pred):
    """
    Compute metrics for model evaluation
    """
    labels = pred.label_ids

    preds = pred.predictions

    if len(preds[0]) >= 2:
        preds = torch.tensor(preds.argmax(-1))
        labels = torch.tensor(labels)

        acc = accuracy_score(labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary"
        )
        return {
            "accuracy": acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }
    else:
        labels = torch.tensor(pred.label_ids[:, np.newaxis])
        preds = torch.tensor(pred.predictions)

        # MSE
        mse = nn.MSELoss()
        mse_loss = mse(labels, preds)

        # RMSE
        rmse = torch.sqrt(mse_loss)

        # MAE
        mae = nn.L1Loss()
        mae_loss = mae(labels, preds)

        return {
            "mse": mse_loss,
            "rmse": rmse,
            "mae": mae_loss,
        }


# Create custom Trainer for training SkipBERT
class SkipBertTrainer(Trainer):
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: Optional[nn.Module] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Callable] = None,
        compute_metrics: Optional[Callable] = None,
        alpha: float = 0.5,
        temperature: float = 2.0,
        beta: float = 1.0,
        use_logits: bool = True,
        use_att: bool = True,
        use_rep: bool = True,
        use_embedding: bool = True,
        att_layer_maps: Optional[List[int]] = None,
        hid_layer_maps: Optional[List[int]] = None,
        epochs_no_cls: int = 0,
        reduce_T: int = 1,
        output_mode: str = "classification",
        num_masked_layers_teacher: int = 0,
        num_masked_last_layers_teacher: int = 0,
        fp16: bool = False,
        num_full_hidden_layers_student: int = 0,
        **kwargs,
    ):
        """
        Initialize SkipBERT Trainer with knowledge distillation capabilities.

        Args:
            student_model: The student model to be trained
            teacher_model: The teacher model for knowledge distillation
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            args: Training arguments
            alpha: Balance between distillation loss and cross-entropy loss
            temperature: Temperature for softening probability distributions
            beta: Weighting factor for different loss components
            use_logits: Whether to use logits-based distillation
            use_att: Whether to use attention-based distillation
            use_rep: Whether to use representation-based distillation
            use_embedding: Whether to use embedding-based distillation
        """
        # Set default training arguments if not provided
        if args is None:
            args = TrainingArguments(
                output_dir="./results",
                num_train_epochs=3,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                logging_dir="./logs",
                evaluation_strategy=EvaluationStrategy.EPOCH,
                save_strategy=IntervalStrategy.EPOCH,
            )

        # Call parent constructor
        super().__init__(
            model=student_model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            **kwargs,
        )

        # Store additional knowledge distillation parameters
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        self.beta = beta
        self.use_logits = use_logits
        self.use_att = use_att
        self.use_rep = use_rep
        self.use_embedding = use_embedding
        self.att_layer_maps = att_layer_maps or []
        self.hid_layer_maps = hid_layer_maps or []
        self.epochs_no_cls = epochs_no_cls
        self.reduce_T = reduce_T
        self.output_mode = output_mode
        self.num_masked_layers_teacher = num_masked_layers_teacher
        self.num_masked_last_layers_teacher = num_masked_last_layers_teacher
        self.num_full_hidden_layers_student = num_full_hidden_layers_student
        self.tr_att_loss = 0
        self.tr_rep_loss = 0
        self.tr_cls_loss = 0
        self.list_att_loss = []
        self.list_rep_loss = []
        self.list_embed_loss = []

        # Prepare FP16 if enabled
        self.fp16 = fp16
        if fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )

            # Initialize amp
            self.model, self.optimizer = amp.initialize(
                self.model, self.optimizer, opt_level="01"
            )

            # Half precision for teacher model if exists
            if self.teacher_model is not None:
                self.teacher_model = self.teacher_model.half()

        # Loss functions
        self.loss_mse = MSELoss()

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        # Separate labels from inputs
        labels = inputs.pop("labels")

        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}

        # Forward pass through student model
        student_logits, student_atts, student_reps = model(**inputs)
        student_reps = student_reps[-self.num_full_hidden_layers_student - 1 :]

        # Forward pass through teacher model
        self.teacher_model.eval()
        with torch.no_grad():
            # teacher_logits, teacher_atts, teacher_reps = self.teacher_model(**inputs)
            teacher_outputs = self.teacher_model(
                **inputs, output_hidden_states=True, output_attentions=True
            )
            teacher_logits, teacher_atts, teacher_reps = (
                teacher_outputs.logits,
                teacher_outputs.attentions,
                teacher_outputs.hidden_states,
            )
            start, end = self.num_masked_layers_teacher, (
                -1 * self.num_masked_layers_teacher
                if self.num_masked_layers_teacher != 0
                else None
            )
            teacher_reps = teacher_reps[start:end]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = student_outputs[self.args.past_index]

        # Compute losses
        att_loss, rep_loss = 0.0, 0.0

        # ---------------------------
        if labels is not None:

            # ---------------------------
            if self.att_layer_maps is None:
                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [
                    teacher_atts[(i * 1) * layers_per_block - 1]
                    for i in range(student_layer_num)
                ]
                assert len(student_atts) == len(new_teacher_atts)

            else:
                new_teacher_atts = []
                for t2s in self.att_layer_maps:
                    if t2s >= 0:
                        new_teacher_atts.append(teacher_atts[t2s])
                    else:
                        new_teacher_atts.append(None)

            # ----------------------------

            for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                if teacher_att is None:
                    continue
                student_att = torch.where(
                    student_att <= 1e-2,
                    torch.zeros_like(student_att),
                    student_att,
                )

                teacher_att = torch.where(
                    teacher_att <= 1e-2,
                    torch.zeros_like(teacher_att),
                    teacher_att,
                )

                att_loss += self.loss_mse(student_att, teacher_att)

            # ---------------------------

            if self.hid_layer_maps is None:
                teacher_layer_num = len(teacher_atts) - 1
                student_layer_num = len(student_atts) - 1
                assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_reps = [
                    teacher_reps[i * layers_per_block]
                    for i in range(student_layer_num + 1)
                ]
                assert len(new_student_reps) == len(new_teacher_reps)
            else:
                new_student_reps = student_reps
                new_teacher_reps = []
                for t2s in self.hid_layer_maps:
                    if t2s >= 0:
                        new_teacher_reps.append(teacher_reps[t2s])
                    else:
                        new_teacher_reps.append(None)

            # ---------------------------

            for student_rep, teacher_rep in zip(
                new_student_reps, new_teacher_reps
            ):
                if teacher_rep is None:
                    continue
                tmp_loss = self.loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss

            self.tr_att_loss += att_loss.item()
            self.tr_rep_loss += rep_loss.item()

            # ---------------------------
            embedding_loss = 0
            if self.use_embedding:
                embedding_loss = self.loss_mse(student_reps[0], teacher_reps[0])

            # ---------------------------

            # ---------------------------

            if self.use_logits and self.state.epoch >= self.epochs_no_cls:
                if isinstance(student_logits, tuple) or isinstance(
                    student_logits, list
                ):
                    cls_loss = None
                    _scale = 0.0
                    for il, logits in enumerate(student_logits):
                        _loss, _, _ = self._compute_distillation_loss(
                            student_logits,
                            student_atts,
                            student_reps,
                            teacher_logits,
                            teacher_atts,
                            teacher_reps,
                            labels,
                        )
                        if cls_loss is None:
                            cls_loss = _loss
                        else:
                            cls_loss = _loss * (il + 1.0) + cls_loss
                        _scale += il + 1

                    cls_loss = cls_loss * (1.0 / _scale)

                else:
                    cls_loss, kd_loss, ce_loss = (
                        self._compute_distillation_loss(
                            student_logits,
                            student_atts,
                            student_reps,
                            teacher_logits,
                            teacher_atts,
                            teacher_reps,
                            labels,
                        )
                    )
                self.tr_cls_loss += cls_loss.item()

            else:
                cls_loss = 0

            # ---------------------------

            check = self.state.epoch >= self.epochs_no_cls
            self.beta = self.beta * check + (1 - check) * 1.0

            # ---------------------------

            if self.use_embedding and self.use_att and self.use_rep:
                loss = (
                    self.beta * (rep_loss + att_loss + embedding_loss)
                    + cls_loss
                )

            elif self.use_att and self.use_rep:
                loss = self.beta * (rep_loss + att_loss) + cls_loss

            elif self.use_embedding and self.use_att:
                loss = self.beta * (att_loss + embedding_loss) + cls_loss

            elif self.use_embedding and self.use_rep:
                loss = self.beta * (rep_loss + embedding_loss) + cls_loss

            elif self.use_att and not self.use_embedding and not self.use_rep:
                loss = self.beta * att_loss + cls_loss

            elif self.use_rep and not self.use_embedding and not self.use_att:
                loss = self.beta * rep_loss + cls_loss

            else:
                loss = cls_loss

            # ---------------------------

        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # ---------------------------

        # ---------------------------
        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
        ):
            loss *= self.accelerator.num_processes
            rep_loss *= self.accelerator.num_processes
            att_loss *= self.accelerator.num_processes
            embedding_loss *= self.accelerator.num_processes
            self.list_att_loss.append(att_loss.item())
            self.list_rep_loss.append(rep_loss.item())
            self.list_embed_loss.append(embedding_loss.item())
        # ---------------------------

        # ---------------------------

        # Ensure logits are properly formatted for evaluation metrics
        logits = student_logits
        if return_outputs:
            # Ensure student_logits has the correct shape [batch_size, num_classes]
            if isinstance(student_logits, (tuple, list)):
                logits = student_logits[-1]
            else:
                logits = student_logits

            # If logits is 1D, reshape it to 2D
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

            # Ensure we have [batch_size, num_classes] shape
            if len(logits.shape) != 2:
                raise ValueError(
                    f"Unexpected logits shape: {logits.shape}. Expected [batch_size, num_classes]"
                )

            if self.output_mode == "classification":  # Classification
                loss = nn.functional.cross_entropy(
                    labels.view(-1),
                    logits.view(-1, len(logits[0])),
                    reduction="mean",
                )

            elif self.output_mode == "regression":  # Regression
                # print(f"Return output -  student: {nn.functional.softmax(student_logits, dim=0).view(-1)}, labels: {labels.view(-1)}")
                loss = self.loss_mse(labels.view(-1), logits.view(-1))

        # ---------------------------
        # print(f"loss: {loss}, att_loss: {att_loss}, rep_loss: {rep_loss}, embed_loss: {embedding_loss}, Train {return_outputs}")
        return (loss, logits) if return_outputs else loss

    def _compute_distillation_loss(
        self,
        student_logits,
        student_atts,
        student_reps,
        teacher_logits,
        teacher_atts,
        teacher_reps,
        labels,
    ):
        """
        Compute comprehensive knowledge distillation loss.

        Args:
            student_*: Student model's outputs
            teacher_*: Teacher model's outputs
            labels: Ground truth labels

        Returns:
            Computed loss
        """

        # Classification/distillation loss
        if self.output_mode == "classification":  # Classification
            # Similar to previous implementation's distillation loss
            if teacher_logits is not None:
                student_likelihood = nn.functional.log_softmax(
                    student_logits / self.temperature, dim=-1
                )
                targets_prob = nn.functional.softmax(
                    teacher_logits / self.temperature, dim=-1
                )
                d_loss = (
                    (-targets_prob * student_likelihood).mean()
                    * (self.temperature**2)
                    / self.reduce_T
                )
            else:
                d_loss = 0
            # Standard cross-entropy/MSE loss
            nll_loss = nn.functional.cross_entropy(
                student_logits, labels, reduction="mean"
            )

        elif self.output_mode == "regression":  # Regression
            # student_likelihood = nn.functional.softmax(student_logits, dim=0)
            # teacher_likelihood = nn.functional.softmax(teacher_logits, dim=0)
            student_likelihood = torch.tensor(student_logits)
            teacher_likelihood = torch.tensor(teacher_logits)
            d_loss = self.loss_mse(
                student_likelihood.view(-1), teacher_likelihood.view(-1)
            )
            nll_loss = self.loss_mse(
                teacher_likelihood.view(-1), labels.view(-1)
            )
        else:
            assert output_mode in ["classification", "regression"]
            d_loss = 0.0
            nll_loss = 0.0
        tol_loss = self.alpha * d_loss + (1 - self.alpha) * nll_loss
        return tol_loss, d_loss, nll_loss

    def train(
        self,
        resume_from_checkpoint: Optional[str] = None,
        trial: Optional[Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Train method with explicit configuration for knowledge distillation training.

        Args:
            resume_from_checkpoint: Optional checkpoint to resume training
            trial: Optional hyperparameter trial configuration
            ignore_keys_for_eval: Keys to ignore during evaluation
        """
        # Prepare teacher model if exists
        if self.teacher_model is not None:
            self.teacher_model.to(self.args.device)
            self.teacher_model.eval()  # Ensure teacher is in eval mode

        # Call parent train method
        return super().train(
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
            **kwargs,
        )

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs = self._prepare_inputs(inputs)

        for param in model.parameters():
            param.requires_grad = True

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(
                model, inputs, self.args.gradient_accumulation_steps
            )
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(
                model, inputs, num_items_in_batch=num_items_in_batch
            )

        del inputs
        if (
            self.args.torch_empty_cache_steps is not None
            and self.state.global_step % self.args.torch_empty_cache_steps == 0
        ):
            if is_torch_xpu_available():
                torch.xpu.empty_cache()
            elif is_torch_mlu_available():
                torch.mlu.empty_cache()
            elif is_torch_musa_available():
                torch.musa.empty_cache()
            elif is_torch_npu_available():
                torch.npu.empty_cache()
            elif is_torch_mps_available(min_version="2.0"):
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()

        kwargs = {}

        # For LOMO optimizers you need to explicitly use the learnign rate
        if self.args.optim in [OptimizerNames.LOMO, OptimizerNames.ADALOMO]:
            kwargs["learning_rate"] = self._get_learning_rate()

        if self.args.n_gpu > 1:
            loss = (
                loss.mean()
            )  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                # scaled_loss.requires_grad = True
                scaled_loss.backward()

            if (
                self.state.global_step + 1
            ) % self.args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(
                    amp.master_params(self.optimizer[0]), 1.0
                )

        else:
            # Finally we need to normalize the loss for reporting
            # loss.requires_grad = True
            if (
                not self.model_accepts_loss_kwargs
                and self.compute_loss_func is None
            ):
                loss = loss / self.args.gradient_accumulation_steps

            # Turning off loss scaling w.r.t. gradient accumulation when DeepSpeed is enabled
            # https://github.com/huggingface/transformers/pull/35808
            if self.accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs["scale_wrt_gas"] = False

            self.accelerator.backward(loss, **kwargs)

            if (
                self.state.global_step + 1
            ) % self.args.gradient_accumulation_steps == 0:
                # nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            return loss.detach()

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        **kwargs,
    ) -> Dict[str, float]:
        """
        Evaluation method with custom metrics computation.

        Args:
            eval_dataset: Optional evaluation dataset
            ignore_keys: Keys to ignore during evaluation
            metric_key_prefix: Prefix for metrics

        Returns:
            Dictionary of evaluation metrics
        """
        # Use parent's evaluate method with optional customizations
        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            **kwargs,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[
        Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]
    ]:
        """
        Override prediction step to handle the model's output format correctly.
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )

        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(
                tuple(inputs.get(name) for name in self.label_names)
            )
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            loss, outputs = self.compute_loss(
                model, inputs, return_outputs=True
            )
            loss = loss.mean().detach()

            # Get logits from outputs
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                # logits = outputs[0]
                logits = outputs

            # Ensure logits has correct shape [batch_size, num_classes]
            if len(logits.shape) == 1:
                logits = logits.unsqueeze(0)

        if prediction_loss_only:
            return (loss, None, None)

        if labels is not None:
            labels = labels.detach()

        logits = nested_detach(logits)
        if len(logits.shape) == 1:
            logits = logits[0]

        # print(f"Validation loss: {loss}")
        return (loss, logits, labels)
