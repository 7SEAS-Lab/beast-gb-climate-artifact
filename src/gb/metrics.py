import torch
from jigsaw.modules import DistributedMSELoss


def df_pm(loss_fn):
    """Calculate data fitting and prior matching values from loss."""
    full_log = loss_fn.log
    loss_fn._init_log()
    full_df = torch.tensor(full_log["data_fitting"], requires_grad=False)
    full_pm = torch.tensor(full_log["prior_matching"], requires_grad=False)
    avg_df = full_df.mean(dim=0).item()
    avg_pm = full_pm.mean(dim=0).item()
    return avg_df, avg_pm


def mse_loss_fn(dtp_group):
    """Return the MSE of the whole data item, sharded over ``dtp_group`` or not.

    With model parallelism, jigsaw's loss reduces over the shards and returns the
    global MSE on every rank. Without it, this is plain ``torch.nn.MSELoss``.
    """
    if dtp_group.size() > 1:
        return DistributedMSELoss(dtp_group)
    return torch.nn.MSELoss()
