import torch
from fancy_einsum import einsum
from src.ActivationCache import record_activations


class RegressionProbe(torch.nn.Module):
    def __init__(self, shape, lr, ridge_alpha=0.0, use_ridge=False):
        super().__init__()
        self.shape = shape
        self.probe_model = torch.nn.Parameter(torch.randn(*shape) * 0.02)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.AdamW([self.probe_model], lr=lr, weight_decay=0.01)
        self.ridge_alpha = ridge_alpha
        self.use_ridge = use_ridge

    def load_weights(self, filepath):
        """Initialize probe weights from a saved file"""
        self.probe_model.data = torch.load(filepath)

    def compute_loss(self, preds_flat, labels_flat):
        """Compute loss with optional ridge regularization"""
        mse_loss = self.loss_fn(preds_flat, labels_flat)

        if self.use_ridge and self.ridge_alpha > 0:
            # Add L2 regularization term
            l2_penalty = self.ridge_alpha * torch.sum(self.probe_model**2)
            return mse_loss + l2_penalty
        else:
            return mse_loss

    def forward(self, x):
        """Produce a single continuous prediction per example."""
        if len(self.shape) == 4:
            input_shape = "modules batch seq d_model"
            probe_shape = "modules seq d_model output_dim"
            output_shape = "modules batch seq"
        elif len(self.shape) == 5:
            input_shape = "modules batch seq n_heads d_model"
            probe_shape = "modules seq n_heads d_model output_dim"
            output_shape = "modules batch seq n_heads"

        else:
            breakpoint()
        x = einsum(
            f"{input_shape}, {probe_shape} -> {output_shape}",
            x,
            self.probe_model,
        )
        return x

    @torch.no_grad()
    def evaluate_probe(self, inputs, labels):
        """Compute MSE per module"""
        self.eval()
        preds = self(inputs)  # (M, B, S) or (M, B, S, H)
        diffs = preds - labels.unsqueeze(0)
        # diffs_per_mod = diffs.reshape(diffs.shape[0], -1) # (M, B*S)
        mse_per_seq = torch.mean(diffs * diffs, dim=1)  # (M, S)
        mae_per_seq = torch.mean(torch.abs(diffs), dim=1)  # (M, S)

        mse_per_module = mse_per_seq.mean(dim=-1)
        mae_per_module = mae_per_seq.mean(dim=-1)
        return (
            mse_per_module.cpu().numpy(),
            mae_per_module.cpu().numpy(),
            mse_per_seq.cpu().numpy(),
            mae_per_seq.cpu().numpy(),
        )


    def train_loop(
        self,
        hooked_model,
        module_names,
        train_prompts,
        train_labels,
        test_activations,
        test_labels,
        input_offset=0,
        label_offset=0,
        epochs=10,
        batch_size=64,
        eval_every=100,
        patience=10,
    ):
        """Train M independent regression probes on raw inputs"""
        M = self.probe_model.shape[0]
        training_losses = []
        validation_mse = []
        validation_mae = []
        best_validation_mae = 9999
        curr_patience = 0

        for epoch in range(epochs):
            for step, idx in enumerate(range(0, len(train_prompts), batch_size)):
                self.train()
                self.optimizer.zero_grad()
                batch_prompts = train_prompts[idx : idx + batch_size]
                batch_labels = train_labels[idx : idx + batch_size]
                with torch.no_grad():
                    with record_activations(hooked_model, module_names) as cache:
                        _ = hooked_model(batch_prompts)

                if input_offset == 0:
                    acts = torch.stack(
                        [cache[m][:, -batch_labels.shape[1] :] for m in module_names],
                        dim=0,
                    )  # (M, batch, seq, n_heads, d_model)
                elif input_offset > 0:
                    acts = torch.stack(
                        [
                            cache[m][:, -batch_labels.shape[1] + input_offset :]
                            for m in module_names
                        ],
                        dim=0,
                    )
                else:
                    acts = torch.stack(
                        [
                            cache[m][:, -batch_labels.shape[1] : input_offset]
                            for m in module_names
                        ],
                        dim=0,
                    )

                if label_offset > 0:
                    batch_labels = batch_labels[:, label_offset:]
                elif label_offset < 0:
                    batch_labels = batch_labels[:, :label_offset]
                preds = self(acts)  # (M, batch, seq)
                preds_flat = preds.reshape(-1)
                if len(self.shape) == 4:
                    labels_flat = (
                        batch_labels.unsqueeze(0).repeat((M, 1, 1)).reshape(-1)
                    )
                elif len(self.shape) == 5:
                    n_heads = acts.shape[3]
                    labels_flat = (
                        batch_labels.unsqueeze(0)
                        .unsqueeze(-1)
                        .repeat((M, 1, 1, n_heads))
                        .reshape(-1)
                    )
                loss = self.loss_fn(preds_flat, labels_flat)
                loss.backward()
                self.optimizer.step()
                training_losses.append(loss.item())

                if step % eval_every == 0:
                    val_mse, val_mae, val_mse_per_seq, val_mae_per_seq = (
                        self.evaluate_probe(test_activations, test_labels)
                    )
                    print(f"Epoch {epoch+1}, Step {step}")
                    print(f"    Val MSE per module: {val_mse.tolist()}")
                    print(f"    Val MAE per module: {val_mae.tolist()}")
                    print(f"    val_mse_per_seq: {val_mse_per_seq}")
                    print(f"    val_mae_per_seq: {val_mae_per_seq}")
                    validation_mse.append(val_mse)
                    validation_mae.append(val_mae)
                    if val_mae.mean() < best_validation_mae:
                        best_validation_mae = val_mae.mean()
                        print(f"New best validation MAE: {best_validation_mae}")
                        curr_patience = 0
                    else:
                        curr_patience += 1
                        if curr_patience >= patience:
                            print("Early stopping triggered.")
                            return training_losses, validation_mse, validation_mae

        return training_losses, validation_mse, validation_mae
