from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from models import StyleEncoder
from utils import NullableArgs, get_model_path


class StyleExtractor:
    """Extracts a style feature vector from a FLAME motion sequence using a trained StyleEncoder."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = 'cuda',
    ) -> None:
        self.device = torch.device(device) if isinstance(device, str) else device

        model_data = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model_args = NullableArgs(model_data['args'])
        self.model = StyleEncoder(self.model_args).to(self.device)
        self.model.encoder.load_state_dict(model_data['encoder'], strict=False)
        self.model.eval()

        stats_file: Path | None = self.model_args.stats_file
        if stats_file is not None:
            if not stats_file.is_absolute():
                stats_file = self.model_args.data_root / stats_file
            coef_stats = dict(np.load(stats_file))
            self.coef_stats: dict[str, torch.Tensor] | None = {
                k: torch.from_numpy(v).to(self.device) for k, v in coef_stats.items()
            }
        else:
            self.coef_stats = None

        self.n_motions: int = self.model_args.n_motions
        self.rot_repr: str = self.model_args.rot_repr
        self.no_head_pose: bool = self.model_args.no_head_pose

    @torch.no_grad()
    def extract(self, coef_file: str | Path, start_frame: int = 0) -> np.ndarray:
        """Extract a style feature from a FLAME coefficient sequence.

        Args:
            coef_file: path to a `.npz` file containing 'exp' and 'pose' arrays
            start_frame: first frame to use (uses `n_motions` frames from here)

        Returns:
            np.ndarray: 1D style feature vector
        """
        end_frame = start_frame + self.n_motions

        coef_raw = dict(np.load(coef_file))
        coef: dict[str, torch.Tensor] = {
            k: torch.from_numpy(coef_raw[k][start_frame:end_frame]).float().to(self.device)
            for k in ['exp', 'pose']
        }

        if self.rot_repr == 'aa':
            coef_keys = ['exp', 'pose']
        else:
            raise ValueError(f'Unknown rotation representation: {self.rot_repr}')

        # Normalize if stats are available
        if self.coef_stats is not None:
            coef = {
                k: (coef[k] - self.coef_stats[f'{k}_mean']) / self.coef_stats[f'{k}_std']
                for k in coef_keys
            }

        if self.no_head_pose:
            if self.rot_repr == 'aa':
                mouth_pose_coef = coef['pose'][:, 3:]
            else:
                raise ValueError(f'Unknown rotation representation: {self.rot_repr}')
            motion_coef = torch.cat([coef['exp'], mouth_pose_coef], dim=-1)
        else:
            motion_coef = torch.cat([coef[k] for k in coef_keys], dim=-1)

        if self.rot_repr == 'aa':
            # Remove mouth rotation around y, z axes
            motion_coef = motion_coef[:, :-2]

        style_feat = self.model(motion_coef.unsqueeze(0))
        return style_feat[0].detach().cpu().numpy()


def main(args: argparse.Namespace) -> None:
    checkpoint_path, exp_name = get_model_path(args.exp_name, args.iter, 'SE')
    extractor = StyleExtractor(checkpoint_path, device=args.device)
    output_dir = Path('demo/input/style') / exp_name / f'iter_{args.iter:07}'

    style_feat = extractor.extract(args.coef, args.start_frame)

    output_file: Path = args.output
    if not output_file.is_absolute():
        output_file = output_dir / output_file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, style_feat)
    print(f'Saved style feature to {output_file}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract a style feature from a FLAME motion sequence using a trained StyleEncoder.'
    )

    # Model
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='style encoder experiment name')
    parser.add_argument('--iter', type=int, default=30000, help='checkpoint iteration')
    parser.add_argument('--device', type=str, default='cuda', help='device (cuda or cpu)')

    # Data
    parser.add_argument('--coef', '-c', type=Path, required=True, help='path to FLAME coefficients (.npz)')
    parser.add_argument('--start_frame', '-s', type=int, default=0, help='starting frame index')
    parser.add_argument('--output', '-o', type=Path, required=True, help='output .npy file name')

    args = parser.parse_args()
    main(args)
