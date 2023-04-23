import os
import torch
import glob


def _checkpoint_path(ckpt_dir: str,
                     step,
                     prefix: str = 'checkpoint_') -> str:
    return os.path.join(ckpt_dir, f'{prefix}{step}.ckpt')


def natural_sort(file_list):
    return sorted(file_list, key=lambda s: float(s[:-4].split('_')[-1]))


def latest_checkpoint(ckpt_dir, prefix: str = 'checkpoint_'):
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    glob_path = os.path.join(ckpt_dir, f'{prefix}*')
    checkpoint_files = natural_sort(glob.glob(glob_path))
    if checkpoint_files:
        return checkpoint_files[-1]
    else:
        return None


def restore_checkpoint(
        ckpt_dir,
        model,
        optimizer=None,
        step=None,
        prefix='checkpoint_'
):
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    if step is not None:
        ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
        if not os.path.exists(ckpt_path):
            raise ValueError(f'Matching checkpoint not found: {ckpt_path}')
    else:
        if not os.path.exists(ckpt_dir):
            print('Found no checkpoint directory at %s' % ckpt_dir)
            return 0
        if not os.path.isdir(ckpt_dir):
            ckpt_path = ckpt_dir
        else:
            ckpt_path = latest_checkpoint(ckpt_dir, prefix)
            if not ckpt_path:
                print('Found no checkpoint files in %s with prefix %s' % (ckpt_dir, prefix))
                return 0

    print('Restoring checkpoint from %s' % ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])
    if optimizer is not None and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt['step']


def save_checkpoint(ckpt_dir,
                    target,
                    step: int,
                    prefix: str = 'checkpoint_',
                    keep: int = 1) -> str:
    ckpt_dir = os.fspath(ckpt_dir)  # Pathlib -> str
    ckpt_path = _checkpoint_path(ckpt_dir, step, prefix)
    base_path = os.path.join(ckpt_dir, prefix)
    checkpoint_files = glob.glob(base_path + '*')

    """Save the checkpoint bytes via file system."""
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if not ckpt_path in checkpoint_files:
        checkpoint_files.append(ckpt_path)

    checkpoint_files = natural_sort(checkpoint_files)
    # Rename once serialization and writing finished.
    print('Saved checkpoint at %s', ckpt_path)
    target['step'] = step
    torch.save(target, ckpt_path)

    # Remove newer checkpoints
    ind = checkpoint_files.index(ckpt_path) + 1
    newer_ckpts = checkpoint_files[ind:]
    checkpoint_files = checkpoint_files[:ind]
    for path in newer_ckpts:
        print('Removing checkpoint at %s', path)
        os.remove(path)

    # Remove old checkpoint files.
    if len(checkpoint_files) > keep:
        old_ckpts = checkpoint_files[:-keep]
        # Note: old_ckpts is sorted from oldest to newest.
        for path in old_ckpts:
            print('Removing checkpoint at %s', path)
            os.remove(path)

    return ckpt_path
