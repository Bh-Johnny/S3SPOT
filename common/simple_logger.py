import os
import json
import logging
import torch
from datetime import datetime

class SimpleLogger:
    def __init__(self, log_dir, log_filename="training.log", history_filename="history.json"):
        """
        Initializes the logger, sets up file handlers, and prepares the history dictionary.
        
        Args:
            log_dir (str): Directory to save logs and checkpoints.
            log_filename (str): Name of the text log file.
            history_filename (str): Name of the JSON file for tracking metrics.
        """
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.log_file_path = os.path.join(self.log_dir, log_filename)
        self.history_file_path = os.path.join(self.log_dir, history_filename)

        # Configure a specific logger for this project to avoid conflicts with root logger
        self.logger = logging.getLogger("ProjectLogger")
        self.logger.setLevel(logging.INFO)
        
        # Avoid adding multiple handlers if the logger is re-initialized
        if not self.logger.handlers:
            # File Handler
            file_handler = logging.FileHandler(self.log_file_path, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
            # Stream Handler (Console)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(stream_handler)
        
        # Dictionary to track training metrics
        self.history = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "cosin_loss_A": [],
            "cosin_loss_B": []
        }

    def _to_float(self, value):
        """Helper to convert PyTorch tensors to standard Python floats."""
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().item()
        return value

    def log(self, message):
        """Logs a message to both the console and the log file."""
        self.logger.info(message)

    def save_history(self):
        """Saves the training history to a JSON file."""
        try:
            with open(self.history_file_path, 'w') as f:
                json.dump(self.history, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save history: {e}")
    
    def update(self, epoch, train_loss, val_loss=None, cosin_loss_A=None, cosin_loss_B=None):
        """
        Updates the history dictionary with metrics from the current epoch and logs them.
        """
        # Convert inputs to float to save memory and ensure JSON serializability
        train_loss = self._to_float(train_loss)
        val_loss = self._to_float(val_loss)
        cosin_loss_A = self._to_float(cosin_loss_A)
        cosin_loss_B = self._to_float(cosin_loss_B)

        self.history["epoch"].append(epoch)
        self.history["train_loss"].append(train_loss)
        
        # Construct log message
        log_parts = [f"Epoch {epoch}: Train Loss = {train_loss:.6f}"]

        if val_loss is not None:
            self.history["val_loss"].append(val_loss)
            log_parts.append(f"Val Loss = {val_loss:.6f}")
        
        if cosin_loss_A is not None:
            self.history["cosin_loss_A"].append(cosin_loss_A)
            log_parts.append(f"Cosin A = {cosin_loss_A:.6f}")
            
        if cosin_loss_B is not None:
            self.history["cosin_loss_B"].append(cosin_loss_B)
            log_parts.append(f"Cosin B = {cosin_loss_B:.6f}")

        # Log the joined message
        self.log(", ".join(log_parts))
        
        # Auto-save history after every update to prevent data loss on crash
        self.save_history()

    def save_checkpoint(self, model, filename='best_model.pt'):
        """
        Saves the model state dictionary.
        
        Args:
            model (torch.nn.Module): The model to save.
            filename (str): The filename for the checkpoint.
        """
        save_path = os.path.join(self.log_dir, filename)
        torch.save(model.state_dict(), save_path)
        self.log(f"Model saved to {save_path}")

    def count_parameters(self, model):
        """
        Counts and logs the number of learnable parameters in the model.
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.log(f"Total Parameters: {total_params:,}")
        self.log(f"Trainable Parameters: {trainable_params:,}")

# Example Usage:
# logger = SimpleLogger(log_dir='./logs')
# logger.count_parameters(my_model)
# logger.update(epoch=1, train_loss=0.5, val_loss=0.4)
# logger.save_checkpoint(my_model)