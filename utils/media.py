from __future__ import annotations

import shlex
import subprocess
from pathlib import Path


def combine_video_and_audio(
    video_file: str | Path,
    audio_file: str | Path,
    output: str | Path,
    quality: int = 17,
    copy_audio: bool = True,
) -> None:
    audio_codec = '-c:a copy' if copy_audio else ''
    cmd = (
        f'ffmpeg -i {video_file} -i {audio_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p '
        f'{audio_codec} -fflags +shortest -y -hide_banner -loglevel error {output}'
    )
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed with return code {result.returncode}')


def combine_frames_and_audio(
    frame_files: str | Path,
    audio_file: str | Path,
    fps: float,
    output: str | Path,
    quality: int = 17,
) -> None:
    cmd = (
        f'ffmpeg -framerate {fps} -i {frame_files} -i {audio_file} -c:v libx264 -crf {quality} '
        f'-pix_fmt yuv420p -c:a copy -fflags +shortest -y -hide_banner -loglevel error {output}'
    )
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed with return code {result.returncode}')


def convert_video(
    video_file: str | Path,
    output: str | Path,
    quality: int = 17,
) -> None:
    cmd = (
        f'ffmpeg -i {video_file} -c:v libx264 -crf {quality} -pix_fmt yuv420p '
        f'-fflags +shortest -y -hide_banner -loglevel error {output}'
    )
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed with return code {result.returncode}')


def reencode_audio(audio_file: str | Path, output: str | Path) -> None:
    cmd = f'ffmpeg -i {audio_file} -y -hide_banner -loglevel error {output}'
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed with return code {result.returncode}')


def extract_frames(
    filename: str | Path,
    output_dir: str | Path,
    quality: int = 1,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f'ffmpeg -i {filename} -qmin 1 -qscale:v {quality} -y -start_number 0 '
        f'-hide_banner -loglevel error {output_dir / "%06d.jpg"}'
    )
    result = subprocess.run(shlex.split(cmd))
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg failed with return code {result.returncode}')
