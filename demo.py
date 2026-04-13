from __future__ import annotations

import argparse
import math
import os
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils
from models import DiffTalkingHead
from utils import NullableArgs
from utils.media import convert_video, reencode_audio

warnings.filterwarnings('ignore', message='PySoundFile failed. Trying audioread instead.')


class Demo:
    def __init__(
        self,
        args: argparse.Namespace,
        load_flame: bool = True,
        load_renderer: bool = True,
    ) -> None:
        self.args = args
        self.load_flame = load_flame
        self.load_renderer = load_renderer
        self.no_context_audio_feat: bool = args.no_context_audio_feat
        self.device = torch.device(args.device)

        # DiffTalkingHead model
        model_path, exp_name = self._get_model_path(args.exp_name, args.iter)
        self.exp_name = exp_name
        self.iter: int = args.iter
        model_data = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model_args = NullableArgs(model_data['args'])
        self.model = DiffTalkingHead(self.model_args, self.device)
        model_data['model'].pop('denoising_net.TE.pe')
        self.model.load_state_dict(model_data['model'], strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.use_indicator: bool = self.model_args.use_indicator
        self.rot_repr: str = self.model_args.rot_repr
        self.predict_head_pose: bool = not self.model_args.no_head_pose
        if self.model.use_style:
            style_dir = Path(self.model_args.style_enc_ckpt)
            style_dir = Path(*style_dir.with_suffix('').parts[-3::2])
            self.style_dir = style_dir

        # Sequence parameters
        self.n_motions: int = self.model_args.n_motions
        self.n_prev_motions: int = self.model_args.n_prev_motions
        self.fps: int = self.model_args.fps
        self.audio_unit: float = 16000.0 / self.fps
        self.n_audio_samples: int = round(self.audio_unit * self.n_motions)
        self.pad_mode: str = self.model_args.pad_mode

        self.coef_stats = dict(np.load(args.coef_stats))
        self.coef_stats = {k: torch.from_numpy(v).to(self.device) for k, v in self.coef_stats.items()}

        # FLAME model (optional heavy dependency — only loaded when requested)
        if self.load_flame:
            from models.flame import FLAME, FLAMEConfig  # noqa: PLC0415

            self.flame = FLAME(FLAMEConfig)
            self.flame.to(self.device)
            self.flame.eval()

        self.default_output_dir = Path('demo/output') / exp_name / f'iter_{self.iter:07}'
        self.save_coef: bool = getattr(args, 'save_coef', False)

        # Renderer (optional heavy dependency — only loaded when requested)
        if self.load_renderer:
            if os.environ.get('CUDA_VISIBLE_DEVICES'):
                os.environ['EGL_DEVICE_ID'] = os.environ['CUDA_VISIBLE_DEVICES'].split(',')[0]
            from psbody.mesh import Mesh  # noqa: PLC0415
            from utils.renderer import MeshRenderer  # noqa: PLC0415

            self.Mesh = Mesh
            self.uv_coords = np.load('models/data/uv_coords.npz')
            self.size: tuple[int, int] = (640, 640)
            self.renderer = MeshRenderer(self.size, black_bg=args.black_bg)

        # Dynamic thresholding
        if args.dynamic_threshold_ratio > 0:
            self.dynamic_threshold: tuple[float, float, float] | None = (
                args.dynamic_threshold_ratio,
                args.dynamic_threshold_min,
                args.dynamic_threshold_max,
            )
        else:
            self.dynamic_threshold = None

    def infer_from_file(
        self,
        audio_path: str | Path,
        coef_path: str | Path,
        out_path: str | Path,
        style_path: str | Path | None = None,
        tex_path: str | Path | None = None,
        n_repetitions: int = 1,
        ignore_global_rot: bool = False,
        cfg_mode: str | None = None,
        cfg_cond: list[str] | None = None,
        cfg_scale: float | list[float] = 1.15,
    ) -> None:
        coef_dict = self.infer_coeffs(
            audio_path, coef_path, style_path, n_repetitions,
            cfg_mode, cfg_cond, cfg_scale, include_shape=True,
        )
        assert self.load_flame, 'FLAME model is not loaded.'
        verts_list = utils.coef_dict_to_vertices(
            coef_dict, self.flame, self.rot_repr, ignore_global_rot=ignore_global_rot
        ).detach().cpu().numpy()

        out_path = Path(out_path)
        if n_repetitions == 1:
            if self.save_coef:
                self.save_coef_file({k: v[0] for k, v in coef_dict.items()}, out_path.with_suffix('.npz'))
            self.render_to_video(verts_list[0], out_path, audio_path, tex_path)
        else:
            for i, verts in enumerate(verts_list):
                out_path_i = out_path.parent / f'{out_path.stem}_{i:03d}{out_path.suffix}'
                if self.save_coef:
                    self.save_coef_file(
                        {k: v[i] for k, v in coef_dict.items()}, out_path_i.with_suffix('.npz')
                    )
                self.render_to_video(verts, out_path_i, audio_path, tex_path)

    @torch.no_grad()
    def infer_coeffs(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        shape_coef: str | Path | np.ndarray | torch.Tensor,
        style_feat: str | Path | np.ndarray | torch.Tensor | None = None,
        n_repetitions: int = 1,
        cfg_mode: str | None = None,
        cfg_cond: list[str] | None = None,
        cfg_scale: float | list[float] = 1.15,
        include_shape: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run the diffusion model and return a dict of motion coefficients.

        Returns:
            dict with keys 'exp', 'pose', (optionally 'shape'), each of shape
            (n_repetitions, L, dim).
        """
        # Preprocess audio
        if isinstance(audio, (str, Path)):
            audio, _ = librosa.load(audio, sr=16000, mono=True)
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).to(self.device)
        assert audio.ndim == 1, 'Audio must be a 1D tensor.'
        audio_mean, audio_std = torch.mean(audio), torch.std(audio)
        audio = (audio - audio_mean) / (audio_std + 1e-5)

        # Preprocess shape coefficient
        if isinstance(shape_coef, (str, Path)):
            shape_coef = np.load(shape_coef)
            if not isinstance(shape_coef, np.ndarray):
                shape_coef = shape_coef['shape']
        if isinstance(shape_coef, np.ndarray):
            shape_coef = torch.from_numpy(shape_coef).float().to(self.device)
        assert shape_coef.ndim <= 2, 'Shape coefficient must be a 1D or 2D tensor.'
        if shape_coef.ndim > 1:
            shape_coef = shape_coef[0]
        original_shape_coef = shape_coef.clone()
        if self.coef_stats is not None:
            shape_coef = (shape_coef - self.coef_stats['shape_mean']) / self.coef_stats['shape_std']
        shape_coef = shape_coef.unsqueeze(0).expand(n_repetitions, -1)

        # Preprocess style feature
        if style_feat is not None:
            assert self.model.use_style
            if isinstance(style_feat, (str, Path)):
                style_feat = Path(style_feat)
                if not style_feat.exists() and not style_feat.is_absolute():
                    style_feat = style_feat.parent / self.style_dir / style_feat.name
                style_feat = np.load(style_feat)
                if not isinstance(style_feat, np.ndarray):
                    style_feat = style_feat['style']
            if isinstance(style_feat, np.ndarray):
                style_feat = torch.from_numpy(style_feat).float().to(self.device)
            assert style_feat.ndim == 1, 'Style feature must be a 1D tensor.'
            style_feat = style_feat.unsqueeze(0).expand(n_repetitions, -1)

        # Divide into synthesis units
        clip_len = int(len(audio) / 16000 * self.fps)
        stride = self.n_motions
        n_subdivision = 1 if clip_len <= self.n_motions else math.ceil(clip_len / stride)

        # Pad audio
        n_padding_audio_samples = self.n_audio_samples * n_subdivision - len(audio)
        n_padding_frames = math.ceil(n_padding_audio_samples / self.audio_unit)
        if n_padding_audio_samples > 0:
            if self.pad_mode == 'zero':
                padding_value = 0.0
            elif self.pad_mode == 'replicate':
                padding_value = audio[-1].item()
            else:
                raise ValueError(f'Unknown pad mode: {self.pad_mode}')
            audio = F.pad(audio, (0, n_padding_audio_samples), value=padding_value)

        if not self.no_context_audio_feat:
            audio_feat = self.model.extract_audio_feature(audio.unsqueeze(0), self.n_motions * n_subdivision)

        coef_list = []
        prev_motion_feat: torch.Tensor | None = None
        prev_audio_feat: torch.Tensor | None = None
        noise: torch.Tensor | None = None
        for i in range(n_subdivision):
            start_idx = i * stride
            end_idx = start_idx + self.n_motions
            indicator = (
                torch.ones((n_repetitions, self.n_motions)).to(self.device)
                if self.use_indicator
                else None
            )
            if indicator is not None and i == n_subdivision - 1 and n_padding_frames > 0:
                indicator[:, -n_padding_frames:] = 0

            if not self.no_context_audio_feat:
                audio_in = audio_feat[:, start_idx:end_idx].expand(n_repetitions, -1, -1)
            else:
                audio_in = audio[
                    round(start_idx * self.audio_unit) : round(end_idx * self.audio_unit)
                ].unsqueeze(0)

            if i == 0:
                motion_feat, noise, prev_audio_feat = self.model.sample(
                    audio_in, shape_coef, style_feat,
                    indicator=indicator, cfg_mode=cfg_mode,
                    cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                    dynamic_threshold=self.dynamic_threshold,
                )
            else:
                motion_feat, noise, prev_audio_feat = self.model.sample(
                    audio_in, shape_coef, style_feat,
                    prev_motion_feat, prev_audio_feat, noise,
                    indicator=indicator, cfg_mode=cfg_mode,
                    cfg_cond=cfg_cond, cfg_scale=cfg_scale,
                    dynamic_threshold=self.dynamic_threshold,
                )
            prev_motion_feat = motion_feat[:, -self.n_prev_motions :].clone()
            prev_audio_feat = prev_audio_feat[:, -self.n_prev_motions :]

            motion_coef = motion_feat
            if i == n_subdivision - 1 and n_padding_frames > 0:
                motion_coef = motion_coef[:, :-n_padding_frames]
            coef_list.append(motion_coef)

        motion_coef = torch.cat(coef_list, dim=1)

        coef_dict = utils.get_coef_dict(
            motion_coef, None, self.coef_stats, self.predict_head_pose, self.rot_repr
        )
        if include_shape:
            coef_dict['shape'] = original_shape_coef[None, None].expand(
                n_repetitions, motion_coef.shape[1], -1
            )
        return coef_dict

    @torch.no_grad()
    def infer_vertices(
        self,
        audio_path: str | Path,
        coef_path: str | Path,
        style_path: str | Path | None = None,
        n_repetitions: int = 1,
        ignore_global_rot: bool = False,
        cfg_mode: str | None = None,
        cfg_cond: list[str] | None = None,
        cfg_scale: float | list[float] = 1.15,
    ) -> torch.Tensor:
        """Generate 3D face vertices from audio.

        Returns:
            torch.Tensor: (n_repetitions, L, 5023, 3)
        """
        assert self.load_flame, 'FLAME model is not loaded.'
        coef_dict = self.infer_coeffs(
            audio_path, coef_path, style_path, n_repetitions,
            cfg_mode, cfg_cond, cfg_scale, include_shape=True,
        )
        return utils.coef_dict_to_vertices(
            coef_dict, self.flame, self.rot_repr, ignore_global_rot=ignore_global_rot
        )

    def save_coef_file(
        self,
        coef: dict[str, torch.Tensor],
        out_path: str | Path,
    ) -> None:
        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.default_output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        coef_np = {k: v.detach().cpu().numpy() for k, v in coef.items()}
        np.savez_compressed(out_path, **coef_np)

    def render_to_video(
        self,
        verts_list: np.ndarray,
        out_path: str | Path,
        audio_path: str | Path | None = None,
        texture: str | Path | np.ndarray | None = None,
    ) -> None:
        """Render a sequence of meshes to a video file.

        Args:
            verts_list: (L, 5023, 3) vertex array
            out_path: output video path
            audio_path: optional audio to combine with the video
            texture: optional texture image or path
        """
        assert self.load_renderer, 'Renderer is not loaded.'
        faces = self.flame.faces_tensor.detach().cpu().numpy()
        if isinstance(texture, (str, Path)):
            texture = cv2.cvtColor(cv2.imread(str(texture)), cv2.COLOR_BGR2RGB)

        out_path = Path(out_path)
        if not out_path.is_absolute():
            out_path = self.default_output_dir / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=out_path.parent)
        writer = cv2.VideoWriter(
            tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size
        )

        center = np.mean(verts_list, axis=(0, 1))
        for verts in tqdm(verts_list, desc='Rendering'):
            mesh = self.Mesh(verts, faces)
            rendered, _ = self.renderer.render_mesh(mesh, center, tex_img=texture, tex_uv=self.uv_coords)
            writer.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        writer.release()

        if audio_path is not None:
            # Re-encode audio to AAC first to avoid sync issues
            tmp_audio_file = tempfile.NamedTemporaryFile('w', suffix='.aac', dir=out_path.parent)
            reencode_audio(audio_path, tmp_audio_file.name)
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', tmp_video_file.name,
                '-i', tmp_audio_file.name,
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-shortest',
                str(out_path),
            ]
            subprocess.run(ffmpeg_cmd, check=True)
            tmp_audio_file.close()
        else:
            convert_video(tmp_video_file.name, out_path)
        tmp_video_file.close()

    @staticmethod
    def _pad_coef(
        coef: torch.Tensor,
        n_frames: int,
        elem_ndim: int = 1,
    ) -> torch.Tensor:
        if coef.ndim == elem_ndim:
            coef = coef[None]
        elem_shape = coef.shape[1:]
        if coef.shape[0] >= n_frames:
            return coef[:n_frames]
        return torch.cat(
            [coef, coef[[-1]].expand(n_frames - coef.shape[0], *elem_shape)], dim=0
        )

    @staticmethod
    def _get_model_path(exp_name: str, iteration: int) -> tuple[Path, Path]:
        exp_root_dir = Path(__file__).parent / 'experiments/DPT'
        exp_dir = exp_root_dir / exp_name
        if not exp_dir.exists():
            exp_dir = next(exp_root_dir.glob(f'{exp_name}*'))
        model_path = exp_dir / f'checkpoints/iter_{iteration:07}.pt'
        return model_path, exp_dir.relative_to(exp_root_dir)


def main(args: argparse.Namespace) -> None:
    demo_app = Demo(args)
    if args.mode == 'interactive':
        try:
            while True:
                audio = input('Enter audio file path: ')
                coef = input('Enter coefficient file path: ')
                scale = float(input('Enter guiding scale (default: 1.15): ') or 1.15)
                tex = input('Enter texture file path (optional): ')
                output = input('Enter output file path: ')
                if not tex or not Path(tex).exists():
                    tex = None
                print('Generating...')
                demo_app.infer_from_file(audio, coef, output, tex_path=tex, cfg_mode=None, cfg_scale=scale)
                print('Done.\n')
        except KeyboardInterrupt:
            print()
    else:
        cfg_cond = (
            demo_app.model.guiding_conditions
            if args.cfg_cond is None
            else args.cfg_cond.split(',')
        )
        cfg_scale = []
        for cond in cfg_cond:
            if cond == 'audio':
                cfg_scale.append(args.scale_audio)
            elif cond == 'style':
                cfg_scale.append(args.scale_style)

        demo_app.infer_from_file(
            args.audio, args.coef, args.output, args.style, args.tex,
            args.n_repetitions, args.no_head, args.cfg_mode, cfg_cond, cfg_scale,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DiffTalkingHead: Speech-Driven 3D Facial Animation using diffusion model'
    )

    # Model
    parser.add_argument('--exp_name', type=str, default='HDTF_TFHP', help='experiment name')
    parser.add_argument('--iter', type=int, default=1000000, help='number of iterations')
    parser.add_argument(
        '--coef_stats',
        type=str,
        default='datasets/HDTF_TFHP/lmdb/stats_train.npz',
        help='path to the coefficient statistics',
    )

    # Inference
    parser.add_argument('--mode', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--black_bg', action='store_true', help='use black background')
    parser.add_argument(
        '--no_context_audio_feat', action='store_true',
        help='use only the current audio feature (no context)',
    )
    parser.add_argument(
        '--dynamic_threshold_ratio', '-dtr', type=float, default=0,
        help='dynamic thresholding ratio (0 to disable)',
    )
    parser.add_argument('--dynamic_threshold_min', '-dtmin', type=float, default=1.0)
    parser.add_argument('--dynamic_threshold_max', '-dtmax', type=float, default=4.0)
    parser.add_argument('--save_coef', action='store_true', help='save generated coefficients')

    args = parser.parse_known_args()[0]
    if args.mode != 'interactive':
        parser.add_argument('--audio', '-a', type=Path, required=True, help='input audio path')
        parser.add_argument('--coef', '-c', type=Path, required=True, help='input shape coefficients path')
        parser.add_argument('--style', '-s', type=Path, help='talking style feature path')
        parser.add_argument('--tex', '-t', type=Path, help='texture image path')
        parser.add_argument('--no_head', action='store_true', help='exclude head pose')
        parser.add_argument('--output', '-o', type=Path, required=True, help='output video path')
        parser.add_argument('--n_repetitions', '-n', type=int, default=1, help='number of repetitions')
        parser.add_argument('--scale_audio', '-sa', type=float, default=1.15, help='audio guiding scale')
        parser.add_argument('--scale_style', '-ss', type=float, default=3, help='style guiding scale')
        parser.add_argument('--cfg_mode', type=str, choices=['incremental', 'independent'])
        parser.add_argument('--cfg_cond', type=str)

    args = parser.parse_args()
    main(args)
