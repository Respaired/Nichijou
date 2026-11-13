# some imports might be stale, this has gone through a lot of iterations over the past few months, sorry for that!

from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
from array import array
from glob import glob
from typing import Any, Optional

import librosa
import nemo.collections.asr as nemo_asr
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import s3fs
import torch
import torch.nn as nn
from datasets import Audio, DatasetDict, load_dataset, load_from_disk
from torch.utils.data import ConcatDataset, DataLoader, Dataset, IterableDataset, get_worker_info

from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def load_tensor(path: Any) -> torch.Tensor:
    if isinstance(path, torch.Tensor):
        return path

    if isinstance(path, np.ndarray):
        return torch.from_numpy(path)

    if isinstance(path, (list, tuple)):
        values = np.asarray(path)

        if values.dtype.kind == 'f':
            values = values.astype(np.float32, copy = False)

        return torch.from_numpy(values)

    if isinstance(path, (bytes, bytearray)):
        try:
            x = torch.load(io.BytesIO(path), map_location = 'cpu', weights_only = True)
        except TypeError:
            x = torch.load(io.BytesIO(path), map_location = 'cpu')
    else:
        try:
            x = torch.load(path, map_location = 'cpu', weights_only = True)
        except TypeError:
            x = torch.load(path, map_location = 'cpu')
        except (AttributeError, RuntimeError, TypeError, ValueError):
            try:
                x = torch.as_tensor(path)
            except Exception as exc:
                raise TypeError(f'expected tensor in {path}, got {type(path)}') from exc

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    if isinstance(x, (list, tuple)):
        x = torch.tensor(x)

    if not isinstance(x, torch.Tensor):
        raise TypeError(f'expected tensor in {path}, got {type(x)}')

    return x


def get_first_split(ds):
    if isinstance(ds, DatasetDict):
        for name in ['train', 'validation', 'valid', 'dev', 'test']:
            if name in ds:
                return ds[name]

        raise ValueError(f'No usable split. Available: {list(ds.keys())}')

    return ds


def _load_tokenizer(tokenizer: Optional[Any]):
    if tokenizer is None:
        return None

    if isinstance(tokenizer, str):
        return AutoTokenizer.from_pretrained(tokenizer, add_bos_token = True, add_eos_token = True)

    return tokenizer


def load_titanet(device: torch.device):
    titanet = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained('nvidia/speakerverification_en_titanet_large')
    titanet = titanet.to(device).eval()

    for p in titanet.parameters():
        p.requires_grad = False

    return titanet


@torch.no_grad()
def extract_speaker_embs_from_batch(titanet: nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    audio = batch['speaker_audio'].to(device)
    lengths = batch['speaker_audio_lengths'].to(device)
    valid = batch['speaker_audio_valid']
    batch_size = audio.shape[0]
    speaker_embs = torch.zeros(batch_size, 192, dtype=torch.float32, device=device)
    valid_indices = valid.nonzero(as_tuple = True)[0]

    if len(valid_indices) == 0:
        return speaker_embs

    audio_valid = audio[valid_indices]
    lengths_valid = lengths[valid_indices]
    _, embs = titanet.forward(input_signal=audio_valid, input_signal_length=lengths_valid)
    speaker_embs[valid_indices] = embs

    return speaker_embs


class TTSDataset(Dataset):
    def __init__(
        self,
        hf_dataset,
        latent_dim: int,
        need_audio_path: bool = False,
        tokenizer: Optional[Any] = None,
        max_text_len: Optional[int] = None,
        latents_dataset = None,
        audio_dataset = None,
        must_merge: bool = False,
        audio_column: str = 'audio',
        latents_column: str = 'latents',
        text_column: str = 'text',
        max_speech_frames: Optional[int] = None,
        key_index_cache_dir: Optional[str] = None
    ):
        self.ds = hf_dataset
        self.latents_ds = latents_dataset
        self.audio_ds = audio_dataset
        self.latent_dim = latent_dim
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.must_merge = must_merge
        self.audio_column = audio_column
        self.latents_column = latents_column
        self.text_column = text_column
        self.max_speech_frames = None if max_speech_frames is None else int(max_speech_frames)
        cache_dir = key_index_cache_dir or os.environ.get(
            'ECHO_TTS_KEY_INDEX_CACHE',
            '~/.cache/echo_tts/key_indices',
        )
        self.key_index_cache_dir = os.path.expanduser(cache_dir)
        self.key_to_index = self._build_key_index(hf_dataset, 'metadata') if self.must_merge else {}
        self.latents_key_to_index = (
            self._build_key_index(latents_dataset, 'latents')
            if latents_dataset is not None
            else {}
        )
        self.audio_key_to_index = (
            self._build_key_index(audio_dataset, 'audio')
            if audio_dataset is not None
            else {}
        )

        if self.must_merge:
            if not self.key_to_index:
                raise ValueError("must_merge=true requires a 'key' column in metadata_ds")

            if self.latents_ds is not None and (not self.latents_key_to_index):
                raise ValueError("must_merge=true requires a 'key' column in latents_ds")

            if self.audio_ds is not None and (not self.audio_key_to_index):
                raise ValueError("must_merge=true requires a 'key' column in audio_ds")

    def _key_index_cache_path(self, hf_dataset, label: str) -> str:
        fingerprint = getattr(hf_dataset, '_fingerprint', None)

        if fingerprint:
            signature = f'hf:{fingerprint}:rows={len(hf_dataset)}:column=key'
        else:
            n = len(hf_dataset)
            sample_positions = sorted({0, n // 2, n - 1}) if n else []
            sampled_keys = []

            for i in sample_positions:
                try:
                    sampled_keys.append(str(hf_dataset[i].get('key', '')))
                except Exception:
                    sampled_keys.append('')

            signature = f'fallback:{type(hf_dataset).__name__}:rows={n}:keys={sampled_keys}'

        digest = hashlib.sha256(signature.encode('utf-8')).hexdigest()[:24]

        return os.path.join(self.key_index_cache_dir, f'{label}_{digest}.pkl')

    def _build_key_index(self, hf_dataset, label: str):
        if (
            hf_dataset is None
            or not hasattr(hf_dataset, 'column_names')
            or 'key' not in hf_dataset.column_names
        ):
            return {}

        cache_path = self._key_index_cache_path(hf_dataset, label)

        try:
            with open(cache_path, 'rb') as f:
                payload = pickle.load(f)

            if (
                isinstance(payload, dict)
                and payload.get('num_rows') == len(hf_dataset)
                and isinstance(payload.get('key_to_index'), dict)
            ):
                index = payload['key_to_index']
                logger.info(
                    'Loaded cached %s key index | rows=%s | entries=%s | cache=%s',
                    label,
                    f'{len(hf_dataset):,}',
                    f'{len(index):,}',
                    cache_path
                )

                return index

            logger.warning('Ignoring stale/invalid %s key-index cache: %s', label, cache_path)
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning(
                'Failed to load %s key-index cache %s (%s: %s); rebuilding',
                label,
                cache_path,
                type(exc).__name__,
                exc
            )

        logger.info('Building %s key index | rows=%s', label, f'{len(hf_dataset):,}')

        index = {}
        offset = 0

        for batch in hf_dataset.select_columns(['key']).iter(batch_size = 100000):
            index.update(((key, offset + i) for i, key in enumerate(batch['key'])))

            offset += len(batch['key'])

            if offset % 1000000 == 0 or offset == len(hf_dataset):
                logger.info('%s key index: %s/%s rows', label, f'{offset:,}', f'{len(hf_dataset):,}')

        os.makedirs(self.key_index_cache_dir, exist_ok = True)

        payload = {
            'num_rows': len(hf_dataset),
            'fingerprint': getattr(hf_dataset, '_fingerprint', None),
            'key_to_index': index
        }
        tmp_path = f'{cache_path}.tmp.{os.getpid()}'

        try:
            with open(tmp_path, 'wb') as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

            os.replace(tmp_path, cache_path)
            logger.info('Saved %s key index cache: %s', label, cache_path)
        except Exception as exc:
            logger.warning(
                'Failed to save %s key-index cache %s (%s: %s)',
                label,
                cache_path,
                type(exc).__name__,
                exc
            )

            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

        return index

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError('requires a tokenizer for on-the-fly text.')

        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = True
        )

        return (enc['input_ids'][0], enc['attention_mask'][0])

    def _fix_latent_shape(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.float()

        if latent.dim() == 1:
            latent = latent.unsqueeze(-1)

        if latent.shape[-1] != self.latent_dim and latent.shape[0] == self.latent_dim:
            latent = latent.T

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f'latent dim mismatch: expected {self.latent_dim}, got {tuple(latent.shape)}')

        return latent

    def _load_latent(self, row: dict) -> torch.Tensor:
        return self._fix_latent_shape(load_tensor(row[self.latents_column]))

    def _merged_keys(self, row: dict) -> Optional[list[str]]:
        merged = row.get('merged', '❌')

        if merged is None or merged == '❌':
            return None

        if not isinstance(merged, str):
            merged = str(merged)

        keys = [key for key in merged.split('➕') if key]

        return keys or None

    def _get_latents_row(self, idx: int, metadata_row: dict):
        if self.latents_ds is None:
            return metadata_row

        key = metadata_row.get('key', None)

        if key is not None and self.latents_key_to_index:
            latent_idx = self.latents_key_to_index.get(key)

            if latent_idx is None:
                raise KeyError(f'metadata key not found in latents dataset: {key}')

            return self.latents_ds[latent_idx]

        if idx >= len(self.latents_ds):
            raise IndexError(f'metadata row {idx} has no key-based latent match and exceeds latents_ds length={len(self.latents_ds)}')

        return self.latents_ds[idx]

    def _load_merged_latent(self, idx: int, metadata_row: dict) -> torch.Tensor:
        keys = self._merged_keys(metadata_row)

        if not keys:
            return self._load_latent(self._get_latents_row(idx, metadata_row))

        latent_source = self.latents_ds if self.latents_ds is not None else self.ds
        latent_key_to_index = self.latents_key_to_index if self.latents_ds is not None else self.key_to_index

        if not latent_key_to_index:
            raise ValueError("merged latent rows require a 'key' column in the latent source dataset")

        latents = []

        for key in keys:
            if key not in latent_key_to_index:
                raise KeyError(f'merged key not found in latents dataset: {key}')

            latents.append(self._load_latent(latent_source[latent_key_to_index[key]]))

        return torch.concatenate(latents)

    def _get_audio_ref(self, idx: int, row: dict):
        if self.audio_ds is None:
            return row.get(self.audio_column, row.get('filename', row.get('audio', None)))

        key = row.get('key', None)

        if key is not None and self.audio_key_to_index:
            audio_idx = self.audio_key_to_index.get(key)

            if audio_idx is None:
                raise KeyError(f'metadata key not found in audio dataset: {key}')

            audio_row = self.audio_ds[audio_idx]
        else:
            if idx >= len(self.audio_ds):
                raise IndexError(f'metadata row {idx} has no key-based audio match and exceeds audio_ds length={len(self.audio_ds)}')

            audio_row = self.audio_ds[idx]

        return audio_row.get(self.audio_column, None)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        dataset_len = len(self.ds)

        if dataset_len == 0:
            raise IndexError('cannot fetch from an empty dataset')

        last_latent_error = None

        for offset in range(dataset_len):
            current_idx = (idx + offset) % dataset_len
            row = self.ds[current_idx]

            try:
                latent = self._load_merged_latent(current_idx, row)
            except Exception as exc:
                last_latent_error = exc
                logger.warning(
                    'Skipping sample with invalid latent | index=%s | error=%s: %s',
                    current_idx,
                    type(exc).__name__,
                    exc
                )
                continue

            if self.max_speech_frames is not None and int(latent.shape[0]) > self.max_speech_frames:
                last_latent_error = ValueError(f'speech latent too long: frames={int(latent.shape[0])}, max_speech_frames={self.max_speech_frames}')
                logger.warning(
                    'Skipping overlength speech latent | index=%s | frames=%s | max_speech_frames=%s',
                    current_idx,
                    int(latent.shape[0]),
                    self.max_speech_frames
                )
                continue

            text = row[self.text_column]
            input_ids, attention_mask = self._tokenize(text)
            attention_mask = attention_mask.to(torch.bool)
            audio_path = None

            if self.need_audio_path:
                audio_path = self._get_audio_ref(current_idx, row)

            return (input_ids, attention_mask, latent, text, audio_path)

        raise RuntimeError('no valid latent samples found in dataset') from last_latent_error


def configure_length_training_stage(cfg: dict, selection_fraction = None) -> None:
    data, training = (cfg['data'], cfg['training'])
    rate = float(data['codec_rate_hz'])

    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('data.codec_rate_hz must be positive and finite')

    frame_limit = math.floor(60.0 * rate)

    if frame_limit < 1:
        raise ValueError('codec rate gives an empty 60-second canvas')

    old_limit = int(data.get('max_latent_length', data['max_audio_seconds'] * rate))
    buckets = data.get('latent_length_buckets', [old_limit])
    data['max_audio_seconds'] = 60.0
    data['max_latent_length'] = frame_limit
    data['latent_length_buckets'] = sorted({int(b) for b in buckets if 0 < int(b) <= frame_limit} | {frame_limit})
    encoder = cfg['model']['text_encoder']
    old_text_limit = int(training.get('length_original_max_text_length', data['max_text_length']))
    position_limit = int(encoder.get('max_position_embeddings', 8192 if encoder.get('kind', 'scratch') == 'scratch' else old_text_limit))
    data['max_text_length'] = min(position_limit, old_text_limit * 8)

    if data['max_text_length'] < 1:
        raise ValueError('text encoder position limit must be positive')

    fraction = float(training.get('length_training_fraction', 0.5) if selection_fraction is None else selection_fraction)

    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('length_training_fraction must be in (0, 1]')

    data.setdefault('silence_latent_path', os.path.join(os.path.dirname(__file__), 'assets', 'silence.pt'))
    training.update(
        length_training_stage = True,
        length_max_samples = 8,
        length_max_seconds = 60.0,
        length_codec_rate_hz=rate,
        length_original_max_text_length=old_text_limit,
        length_training_fraction=fraction,
        length_silence_latent_path=data['silence_latent_path']
    )


def concatenate_processed_texts(texts):
    leading_tag = re.compile('^\\s*(<[^<>]+>)')
    speaker_tag = re.compile('<S[0-9]+>')
    parts = []

    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            raise ValueError('missing processed transcript')

        text = text.strip()

        if index:
            kept = []

            while (match := leading_tag.match(text)) is not None:
                tag = match.group(1)

                if tag.startswith('<|'):
                    break

                if speaker_tag.fullmatch(tag):
                    kept.append(tag)

                text = text[match.end():]

            text = ''.join(kept) + (' ' if kept and text.strip() else '') + text.lstrip()

        parts.append(text)

    current = None

    def preserve_turn(match):
        nonlocal current
        tag = match.group(0)

        if tag == current:
            return ' '

        current = tag

        return tag
    return re.sub('\\s+', ' ', speaker_tag.sub(preserve_turn, ' '.join(parts))).strip()
_LENGTH_PLAN_STATE = None
_LENGTH_PLAN_TEXT_MAP = None


def _init_length_plan_worker(directory, tokenizer, options):
    global _LENGTH_PLAN_STATE, _LENGTH_PLAN_TEXT_MAP
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    torch.set_num_threads(1)
    pa.set_cpu_count(1)

    arrays = {name: np.load(os.path.join(directory, name + '.npy'), mmap_mode = 'r') for name in ('order', 'ends', 'counts', 'durations')}
    _LENGTH_PLAN_TEXT_MAP = pa.memory_map(os.path.join(directory, 'texts.arrow'), 'r')
    texts = pa.ipc.open_file(_LENGTH_PLAN_TEXT_MAP).read_all()['text']
    _LENGTH_PLAN_STATE = (arrays, texts, tokenizer, options)


def _plan_length_partition(task, state = None):
    arrays, texts, tokenizer, options = _LENGTH_PLAN_STATE if state is None else state
    order, ends, counts, durations = (arrays[k] for k in ('order', 'ends', 'counts', 'durations'))
    gap_frames, gap_seconds, frame_limit, seconds_limit, text_limit = options
    claimed = np.zeros(len(counts), dtype=bool)
    flat, offsets = (array('q'), array('q', [0]))
    frame_totals, duration_totals, ranks = (array('q'), array('d'), array('q'))
    last_report = time.monotonic()

    for processed, (rank, start) in enumerate(task, 1):
        if time.monotonic() - last_report >= 30:
            logger.warning(
                'Length grouping worker %d: %s/%s assigned starts checked',
                os.getpid(),
                f'{processed:,}',
                f'{len(task):,}'
            )

            last_report = time.monotonic()

        start = int(start)
        first = int(order[start])

        if claimed[first] or counts[first] <= 0:
            continue

        rows, parts = ([], [])
        frames, duration = (0, 0.0)

        for position in range(start, int(ends[start])):
            row = int(order[position])

            if claimed[row] or counts[row] <= 0:
                break

            extra_frames = int(counts[row]) + (gap_frames if rows else 0)
            extra_duration = float(durations[row]) + (gap_seconds if rows else 0.0)

            if frames + extra_frames > frame_limit or duration + extra_duration > seconds_limit:
                break

            text = texts[row].as_py()

            if not isinstance(text, str) or not text.strip():
                break

            rows.append(row)
            parts.append(text)

            frames += extra_frames
            duration += extra_duration

        while len(rows) >= 2:
            joined = concatenate_processed_texts(parts)
            ids = tokenizer(joined, truncation = False, padding = False, add_special_tokens = True)['input_ids']
            token_count = ids.shape[-1] if isinstance(ids, torch.Tensor) else len(ids)

            if token_count <= text_limit:
                break

            removed = rows.pop()
            parts.pop()

            frames -= int(counts[removed]) + gap_frames
            duration -= float(durations[removed]) + gap_seconds

        if len(rows) < 2:
            continue

        claimed[rows] = True
        flat.extend(rows)
        offsets.append(len(flat))
        frame_totals.append(frames)
        duration_totals.append(duration)
        ranks.append(rank)

    return tuple((np.asarray(values, dtype=dtype) for values, dtype in ((flat, np.int64), (offsets, np.int64), (frame_totals, np.int64), (duration_totals, np.float64), (ranks, np.int64))))


def numbered_neighbour_order(metadata, max_samples = 8):
    columns = ['key'] + (['episode'] if 'episode' in metadata.column_names else [])
    table = metadata.select_columns(columns).with_format('arrow')[:]

    if len(table) != len(metadata):
        raise AssertionError('metadata selection lost row alignment')

    keys = pc.cast(table['key'], pa.string())
    parsed = pc.extract_regex(
        keys,
        '^(?P<prefix>.+?(?:_chunk|_))(?P<number>[0-9]{1,18})(?:\\.[A-Za-z0-9]+)?$'
    )
    prefix = pc.struct_field(parsed, 'prefix')
    numbers = pc.cast(pc.struct_field(parsed, 'number'), pa.int64())

    if 'episode' in columns:
        episode = pc.fill_null(pc.cast(table['episode'], pa.string()), '')
        prefix = pc.binary_join_element_wise(prefix, episode, '\x00')

    sorting = pa.table({'namespace': prefix, 'number': numbers, 'row': pa.array(np.arange(len(metadata), dtype=np.int64))})
    permutation = pc.sort_indices(
        sorting,
        sort_keys=[('namespace', 'ascending'), ('number', 'ascending'), ('row', 'ascending')]
    )
    sorting = sorting.take(permutation)
    order = sorting['row'].to_numpy().astype(np.int64, copy = False)
    n = len(order)

    if not n:
        return (order, np.empty(0, dtype=np.int64))

    numeric = pc.fill_null(sorting['number'], -1).to_numpy()
    same = pc.fill_null(
        pc.equal(sorting['namespace'].slice(0, n - 1), sorting['namespace'].slice(1)),
        False
    ).to_numpy()
    repeated = same & (numeric[:-1] == numeric[1:])
    duplicate = np.zeros(n, dtype=bool)
    duplicate[:-1] |= repeated
    duplicate[1:] |= repeated
    joins = same & (numeric[1:] == numeric[:-1] + 1) & ~duplicate[:-1] & ~duplicate[1:]
    boundaries = np.r_[np.flatnonzero(~joins), n - 1]
    positions = np.arange(n, dtype=np.int64)
    ends = np.minimum(boundaries[np.searchsorted(boundaries, positions)] + 1, positions + int(max_samples))

    return (order, ends.astype(np.int64, copy = False))


class LengthTrainingDataset(TTSDataset):

    def __init__(
        self,
        hf_dataset,
        *args,
        codec_rate_hz,
        max_concat_samples = 8,
        max_concat_seconds = 60.0,
        selection_fraction = 0.5,
        seed = 42,
        silence_latent_path = None,
        planning_workers = 1,
        **kwargs
    ):
        if kwargs.get('must_merge', False):
            raise ValueError('length training expects individual samples; must_merge=true is unsupported')

        self.codec_rate_hz = float(codec_rate_hz)
        self.max_concat_samples = int(max_concat_samples)
        self.max_concat_seconds = float(max_concat_seconds)
        self.selection_fraction, self.seed = (float(selection_fraction), int(seed))
        self.planning_workers = int(planning_workers)

        if self.planning_workers < 1:
            raise ValueError('length planning_workers must be positive')

        if not math.isfinite(self.codec_rate_hz) or self.codec_rate_hz <= 0:
            raise ValueError('length-training codec rate must be positive and finite')

        if not 2 <= self.max_concat_samples <= 8 or not 0 < self.max_concat_seconds <= 60:
            raise ValueError('length-training limits must be <=8 samples and <=60 seconds')

        if not math.isfinite(self.selection_fraction) or not 0 < self.selection_fraction <= 1:
            raise ValueError('length-training fraction must be in (0, 1]')

        original_latents = kwargs.get('latents_dataset')
        self.inline_latents = original_latents is None or original_latents is hf_dataset
        original_latents = hf_dataset if original_latents is None else original_latents
        latent_column = kwargs.get('latents_column', 'latents')
        kwargs['latents_dataset'] = original_latents.select_columns([c for c in ('key', 'filename', latent_column) if c in original_latents.column_names])
        original_audio = kwargs.get('audio_dataset')
        self.inline_audio = original_audio is hf_dataset

        if original_audio is not None:
            kwargs['audio_dataset'] = original_audio.select_columns(list(dict.fromkeys((c for c in ('key', 'filename', kwargs.get('audio_column', 'audio')) if c in original_audio.column_names))))

        columns = ['key', 'filename', 'duration', 'episode', kwargs.get('text_column', 'text')]

        if kwargs.get('need_audio_path') and original_audio is None:
            columns.append(kwargs.get('audio_column', 'audio'))

        hf_dataset = hf_dataset.select_columns(list(dict.fromkeys((c for c in columns if c in hf_dataset.column_names))))
        super().__init__(hf_dataset, *args, **kwargs)

        if 'key' not in self.ds.column_names or self.tokenizer is None:
            raise ValueError('length training requires keys and a tokenizer')

        frame_limit = math.floor(self.max_concat_seconds * self.codec_rate_hz)
        self.max_speech_frames = min(self.max_speech_frames or frame_limit, frame_limit)
        silence_path = os.path.abspath(silence_latent_path or os.path.join(os.path.dirname(__file__), 'assets', 'silence.pt'))
        silence = torch.load(silence_path, map_location = 'cpu', weights_only = True)

        if isinstance(silence, dict):
            candidates = [silence[k] for k in ('latents', 'latent', 'silence_latent') if k in silence]

            if not candidates:
                candidates = [v for v in silence.values() if isinstance(v, torch.Tensor)]

            if len(candidates) != 1:
                raise ValueError('cannot identify Fish silence latent')

            silence = candidates[0]

        silence = torch.as_tensor(silence).float()

        while silence.ndim > 2 and silence.shape[0] == 1:
            silence = silence[0]

        if silence.ndim == 2 and silence.shape[-1] != self.latent_dim and (silence.shape[0] == self.latent_dim):
            silence = silence.T

        if silence.ndim != 2 or silence.shape[-1] != self.latent_dim or (not len(silence)) or (not torch.isfinite(silence).all()):
            raise ValueError('invalid Fish silence latent')

        self.gap_frames = max(1, int(math.floor(0.25 * self.codec_rate_hz + 0.5)))
        self.gap_seconds = self.gap_frames / self.codec_rate_hz
        self.silence_gap = silence[torch.arange(self.gap_frames) % len(silence)].contiguous()
        self._latent_filenames = self._logical_column(
            self.latents_ds,
            'filename'
        ) if 'filename' in self.latents_ds.column_names else None
        vocab = self.tokenizer.get_vocab() if hasattr(self.tokenizer, 'get_vocab') else type(self.tokenizer).__name__
        signature = dict(
            version = 3,
            fraction=self.selection_fraction,
            seed=self.seed,
            samples=self.max_concat_samples,
            seconds=self.max_concat_seconds,
            frame_limit=self.max_speech_frames,
            rate=self.codec_rate_hz,
            gap=self.gap_frames,
            text_limit=self.max_text_len,
            text_column=self.text_column,
            tokenizer=vocab,
            latent_store=getattr(self.latents_ds, '_fingerprint', None)
        )
        digest = hashlib.sha256(json.dumps(signature, sort_keys = True).encode()).hexdigest()[:24]
        self.group_cache_path = self._key_index_cache_path(self.ds, f'length_groups_v3_{digest}') + '.npz'
        self.loaded_group_cache = False

        try:
            with np.load(self.group_cache_path, allow_pickle = False) as saved:
                self.group_rows = saved['rows']
                self.group_offsets = saved['offsets']
                self.group_frames = saved['frames']
                self.group_durations = saved['durations']

            self.loaded_group_cache = True
        except FileNotFoundError:
            self._plan_groups()
            os.makedirs(os.path.dirname(self.group_cache_path), exist_ok = True)

            temp = f'{self.group_cache_path}.tmp.{os.getpid()}.npz'
            np.savez(
                temp,
                rows=self.group_rows,
                offsets=self.group_offsets,
                frames=self.group_frames,
                durations=self.group_durations
            )
            os.replace(temp, self.group_cache_path)

        self._validate_cached_groups()

        logger.info(
            'Length training: %s original rows -> %s grouped epoch items; %.1f%% random starts; Fish silence gap=%d frames (%.1fms); grouping %s: %s',
            f'{len(self.ds):,}',
            f'{len(self):,}',
            100 * self.selection_fraction,
            self.gap_frames,
            1000 * self.gap_seconds,
            'loaded' if self.loaded_group_cache else 'built',
            self.group_cache_path
        )

    def _validate_cached_groups(self):
        offsets = self.group_offsets
        frames = self.group_frames
        durations = self.group_durations

        if offsets.ndim != 1 or not len(offsets):
            raise ValueError('Cached group offsets must be a nonempty one-dimensional array')

        if frames.ndim != 1 or durations.ndim != 1:
            raise ValueError('Cached group metadata must be one-dimensional')

        group_count = len(offsets) - 1

        if offsets[0] != 0 or offsets[-1] != len(self.group_rows):
            raise ValueError('Cached group offsets do not match the stored rows')

        if len(frames) != group_count or len(durations) != group_count:
            raise ValueError('Cached group metadata lengths do not match the group count')

        group_sizes = np.diff(offsets)

        if np.any(group_sizes < 2):
            raise ValueError('Cached groups must contain at least two samples')

        if np.any(group_sizes > self.max_concat_samples):
            raise ValueError('Cached groups exceed the sample limit')

        if np.any(frames > self.max_speech_frames):
            raise ValueError('Cached groups exceed the speech frame limit')

        if np.any(durations > self.max_concat_seconds + 1e-6):
            raise ValueError('Cached groups exceed the duration limit')

    @staticmethod
    def _logical_column(store, name):
        column = store.data.column(name)

        if getattr(store, '_indices', None) is not None:
            column = pc.take(column, store._indices.column(0))

        return column

    def _matching_latent_index(self, index, row):
        if self.inline_latents:
            return index

        candidates = self.latents_key_to_index.get(row['key'])

        if candidates is None:
            return None

        if isinstance(candidates, int):
            if row.get('filename') is not None and self._latent_filenames is not None:
                if self._latent_filenames[candidates].as_py() != row['filename']:
                    return None

            return candidates

        if row.get('filename') is None or self._latent_filenames is None:
            return None

        for candidate in reversed(candidates):
            if self._latent_filenames[candidate].as_py() == row['filename']:
                return candidate

        return None

    def _plan_groups(self):
        logger.info('Length planning: sorting numbered neighbors for %s rows', f'{len(self.ds):,}')

        order, ends = numbered_neighbour_order(self.ds, self.max_concat_samples)
        n = len(self.ds)
        raw_counts = pc.fill_null(pc.list_value_length(self.latents_ds.data.column(self.latents_column)), 0)

        if getattr(self.latents_ds, '_indices', None) is not None:
            raw_counts = pc.take(raw_counts, self.latents_ds._indices.column(0))

        store_counts = raw_counts.to_numpy().astype(np.int64, copy = False)
        logger.info('Length planning: matching %s metadata rows to codec lengths', f'{n:,}')

        if self.inline_latents:
            latent_indices = np.arange(n, dtype=np.int64)
        else:
            keys = self._logical_column(self.ds, 'key')
            store_keys = self._logical_column(self.latents_ds, 'key')
            latent_indices = pc.fill_null(pc.index_in(keys, value_set=store_keys), -1).to_numpy().copy()
            duplicate_keys = [
                key for key,
                value in self.latents_key_to_index.items() if not isinstance(value, int)
            ]

            if duplicate_keys:
                duplicate_mask = pc.is_in(keys, value_set=pa.array(duplicate_keys)).to_numpy()
                duplicate_rows = np.flatnonzero(duplicate_mask)
                logger.info(
                    'Length planning: resolving %s duplicate-key rows by filename',
                    f'{len(duplicate_rows):,}'
                )

                filenames = self._logical_column(
                    self.ds,
                    'filename'
                ) if 'filename' in self.ds.column_names else None

                for index in duplicate_rows:
                    row = {'key': keys[index].as_py()}

                    if filenames is not None:
                        row['filename'] = filenames[index].as_py()

                    match = self._matching_latent_index(int(index), row)
                    latent_indices[index] = -1 if match is None else match

            if 'filename' in self.ds.column_names and self._latent_filenames is not None:
                filenames = self._logical_column(self.ds, 'filename')
                matched = pc.take(self._latent_filenames, pa.array(latent_indices, mask=latent_indices < 0))
                valid_filename = pc.or_(
                    pc.is_null(filenames),
                    pc.fill_null(pc.equal(filenames, matched), False)
                ).to_numpy()
                latent_indices[~valid_filename] = -1

        counts = np.zeros(n, dtype=np.int64)
        valid = latent_indices >= 0
        counts[valid] = store_counts[latent_indices[valid]]
        counts[counts < 0] = 0
        durations = counts.astype(np.float64) / self.codec_rate_hz

        if 'duration' in self.ds.column_names:
            measured_column = pc.cast(self._logical_column(self.ds, 'duration'), pa.float64())
            measured = measured_column.to_numpy()
            present = ~pc.is_null(measured_column).to_numpy()
            invalid = present & (~np.isfinite(measured) | (measured <= 0))
            counts[invalid] = 0
            durations[present & ~invalid] = np.maximum(durations[present & ~invalid], measured[present & ~invalid])

        logger.info(
            'Length planning: codec-length matching complete; %s valid rows',
            f'{np.count_nonzero(counts):,}'
        )

        texts = self._logical_column(self.ds, self.text_column)
        starts = np.random.default_rng(self.seed).permutation(n)[:int(n * self.selection_fraction)]
        arrays = dict(order=order, ends=ends, counts=counts, durations=durations)
        options = (
            self.gap_frames,
            self.gap_seconds,
            self.max_speech_frames,
            self.max_concat_seconds,
            self.max_text_len
        )
        workers = min(self.planning_workers, os.cpu_count() or 1, max(len(starts), 1))

        if workers == 1:
            logger.info('Length grouping: checking %s starts with one worker', f'{len(starts):,}')

            result = _plan_length_partition(
                np.column_stack((np.arange(len(starts)), starts)),
                (arrays, texts, self.tokenizer, options)
            )
            self.group_rows, self.group_offsets, self.group_frames, self.group_durations, _ = result

            return

        run_ends = ends == np.arange(n, dtype=np.int64) + 1
        run_ids = np.r_[0, np.cumsum(run_ends[:-1], dtype=np.int64)]
        selected_runs, selected_counts = np.unique(run_ids[starts], return_counts = True)
        assignments = np.zeros(len(selected_runs), dtype=np.int64)
        loads = np.zeros(workers, dtype=np.int64)

        for index in np.argsort(-selected_counts, kind = 'stable'):
            worker = int(np.argmin(loads))
            assignments[index] = worker
            loads[worker] += selected_counts[index]

        owners = assignments[np.searchsorted(selected_runs, run_ids[starts])]
        logger.info(
            'Length grouping: %s random starts across %d CPU workers; shared metadata, no model copies',
            f'{len(starts):,}',
            workers
        )

        results = []

        with tempfile.TemporaryDirectory(prefix = 'nichijou_length_plan_') as directory:
            for name, values in arrays.items():
                np.save(os.path.join(directory, name + '.npy'), values)

            table = pa.table({'text': texts})

            with pa.OSFile(os.path.join(directory, 'texts.arrow'), 'wb') as output:
                with pa.ipc.new_file(output, table.schema) as writer:
                    writer.write_table(table)

            np.save(
                os.path.join(directory, 'tasks.npy'),
                np.column_stack((np.arange(len(starts)), starts, owners))
            )

            with open(os.path.join(directory, 'tokenizer.pkl'), 'wb') as output:
                pickle.dump((self.tokenizer, options), output)

            environment = os.environ.copy()
            environment.update(
                CUDA_VISIBLE_DEVICES = '',
                OMP_NUM_THREADS = '1',
                MKL_NUM_THREADS = '1',
                OPENBLAS_NUM_THREADS = '1',
                TOKENIZERS_PARALLELISM = 'false'
            )

            environment['PYTHONPATH'] = os.pathsep.join((os.path.abspath(path or os.getcwd()) for path in sys.path))
            command = [
                sys.executable,
                os.path.join(os.path.dirname(__file__), 'length_planning.py'),
                directory,
                str(workers)
            ]

            with subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, text = True) as process:
                for line in process.stdout:
                    logger.info('%s', line.rstrip())

                if process.wait() != 0:
                    raise RuntimeError('CPU length-planning workers failed; no group cache was saved')

            for worker in range(workers):
                path = os.path.join(directory, f'result_{worker}.npz')

                if os.path.isfile(path):
                    with np.load(path, allow_pickle = False) as result:
                        results.append(tuple((result[name] for name in ('rows', 'offsets', 'frames', 'durations', 'ranks'))))

        flat = np.concatenate([r[0] for r in results])
        lengths = np.concatenate([np.diff(r[1]) for r in results])
        frames = np.concatenate([r[2] for r in results])
        durations = np.concatenate([r[3] for r in results])
        ranks = np.concatenate([r[4] for r in results])
        permutation = np.argsort(ranks)
        old_offsets = np.r_[0, np.cumsum(lengths)]
        self.group_offsets = np.r_[0, np.cumsum(lengths[permutation])]
        self.group_rows = np.empty(len(flat), dtype=np.int64)

        for destination, source in enumerate(permutation):
            self.group_rows[self.group_offsets[destination]:self.group_offsets[destination + 1]] = flat[old_offsets[source]:old_offsets[source + 1]]

        self.group_frames = frames[permutation]
        self.group_durations = durations[permutation]

    def __len__(self):
        return len(self.group_offsets) - 1

    def _build_key_index(self, hf_dataset, label):
        if hf_dataset is None or hf_dataset is self.ds or (label == 'latents' and self.inline_latents) or (label == 'audio' and self.inline_audio):
            return {}

        if 'key' not in hf_dataset.column_names:
            raise ValueError('external stores need keys for length-training joins')

        path = self._key_index_cache_path(hf_dataset, f'length_v1_{label}')

        try:
            with open(path, 'rb') as stream:
                saved = pickle.load(stream)

            if saved.get('version') == 1 and saved.get('num_rows') == len(hf_dataset):
                logger.info('Length planning: loaded %s key index (%s rows)', label, f'{len(hf_dataset):,}')

                return saved['key_to_index']
        except FileNotFoundError:
            pass

        index, offset = ({}, 0)
        logger.info('Length planning: building %s key index for %s rows', label, f'{len(hf_dataset):,}')

        for batch in hf_dataset.select_columns(['key']).iter(batch_size = 100000):
            for row, key in enumerate(batch['key'], offset):
                if key in index:
                    previous = index[key]

                    if isinstance(previous, list):
                        previous.append(row)
                    else:
                        index[key] = [previous, row]
                else:
                    index[key] = row

            offset += len(batch['key'])
            logger.info(
                'Length planning: %s key index %s/%s rows',
                label,
                f'{offset:,}',
                f'{len(hf_dataset):,}'
            )

        os.makedirs(os.path.dirname(path), exist_ok = True)

        temp = f'{path}.tmp.{os.getpid()}'

        with open(temp, 'wb') as stream:
            pickle.dump(
                dict(version = 1, num_rows=len(hf_dataset), key_to_index=index),
                stream,
                protocol=pickle.HIGHEST_PROTOCOL
            )

        os.replace(temp, path)

        return index

    def _exact_store_row(self, store, index, row, key_index):
        if store is None or store is self.ds:
            return row if store is None else store[index]

        candidates = key_index.get(row['key'])

        if candidates is None:
            raise KeyError(f"missing backing-store key: {row['key']}")

        if isinstance(candidates, int):
            result = store[candidates]

            if 'filename' in row and 'filename' in result and (row['filename'] != result['filename']):
                raise ValueError(f"key/filename mismatch: {row['key']}")

            return result

        if row.get('filename') is None or 'filename' not in store.column_names:
            raise ValueError(f"ambiguous backing-store key: {row['key']}")

        filenames = store.select_columns(['filename'])

        for candidate in reversed(candidates):
            if filenames[candidate]['filename'] == row['filename']:
                return store[candidate]

        raise KeyError(f"no filename match: {row['key']}")

    def _get_latents_row(self, idx, row):
        index = self._matching_latent_index(idx, row)

        if index is None:
            raise KeyError(f"no exact latent match for {row['key']}")

        return self.latents_ds[index]

    def _get_audio_ref(self, idx, row):
        if self.audio_ds is None:
            return super()._get_audio_ref(idx, row)

        result = self.audio_ds[idx] if self.inline_audio else self._exact_store_row(self.audio_ds, idx, row, self.audio_key_to_index)

        return result.get(self.audio_column)

    def __getitem__(self, idx):
        if not 0 <= int(idx) < len(self):
            raise IndexError(idx)

        rows = self.group_rows[self.group_offsets[idx]:self.group_offsets[idx + 1]]
        latents, texts, audio = ([], [], [])

        for part_index, row_index in enumerate(rows):
            row_index = int(row_index)
            row = self.ds[row_index]
            source = self._get_latents_row(row_index, row)
            value = source[self.latents_column]

            if value is None:
                raise ValueError(f"cached group contains missing codec latent: {row['key']}")

            if isinstance(value, (list, tuple, np.ndarray)):
                value = torch.from_numpy(np.asarray(value, dtype=np.float32))
            else:
                value = load_tensor(value)

            latent = self._fix_latent_shape(value)

            if latent.ndim != 2 or not len(latent) or (not torch.isfinite(latent).all()):
                raise ValueError(f"invalid latent in cached group: {row['key']}")

            if part_index:
                latents.append(self.silence_gap)

                if self.need_audio_path:
                    audio.append({'array': np.zeros(round(16000 * self.gap_seconds), dtype=np.float32), 'sampling_rate': 16000})

            latents.append(latent)
            texts.append(row[self.text_column])

            if self.need_audio_path:
                audio.append(self._get_audio_ref(row_index, row))

        latent = torch.cat(latents, dim = 0)

        if len(latent) != self.group_frames[idx] or len(latent) > self.max_speech_frames:
            raise ValueError('codec frame count differs from cached grouping; rebuild its index')

        text = concatenate_processed_texts(texts)
        tokens = self.tokenizer(
            text,
            truncation = False,
            padding = False,
            return_tensors = 'pt',
            add_special_tokens = True
        )

        if tokens['input_ids'].shape[1] > self.max_text_len:
            raise ValueError('transcript differs from cached text budget; rebuild its index')

        return (
            tokens['input_ids'][0],
            tokens['attention_mask'][0].bool(),
            latent,
            text,
            audio if self.need_audio_path else None
        )


class MultiSourceStreamingTTSDataset(IterableDataset):

    def __init__(
        self,
        metadata_dataset,
        latents_dataset,
        latent_dim: int,
        need_audio_path: bool = False,
        tokenizer: Optional[Any] = None,
        max_text_len: Optional[int] = None,
        audio_dataset = None,
        audio_column: str = 'audio',
        latents_column: str = 'latents',
        text_column: str = 'text',
        selected_indices: Optional[set[int]]=None,
        max_speech_frames: Optional[int] = None
    ):
        self.metadata_ds = metadata_dataset
        self.latents_ds = latents_dataset
        self.audio_ds = audio_dataset
        self.latent_dim = latent_dim
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.audio_column = audio_column
        self.latents_column = latents_column
        self.text_column = text_column
        self.selected_indices = selected_indices
        self.max_speech_frames = None if max_speech_frames is None else int(max_speech_frames)

    def __len__(self):
        if self.selected_indices is not None:
            return len(self.selected_indices)

        return len(self.metadata_ds)

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError('requires a tokenizer for on-the-fly text.')

        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = True
        )

        return (enc['input_ids'][0], enc['attention_mask'][0])

    def _fix_latent_shape(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.float()

        if latent.dim() == 1:
            latent = latent.unsqueeze(-1)

        if latent.shape[-1] != self.latent_dim and latent.shape[0] == self.latent_dim:
            latent = latent.T

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f'latent dim mismatch: expected {self.latent_dim}, got {tuple(latent.shape)}')

        return latent

    def _row_to_item(
        self,
        metadata_row: dict,
        latents_row: dict,
        audio_row: Optional[dict],
        row_index: Optional[int]=None
    ):
        try:
            latent = self._fix_latent_shape(load_tensor(latents_row[self.latents_column]))
        except Exception as exc:
            logger.warning(
                'Skipping streamed sample with invalid latent | index=%s | key=%s | error=%s: %s',
                row_index,
                latents_row.get('key'),
                type(exc).__name__,
                exc
            )

            return None

        if self.max_speech_frames is not None and int(latent.shape[0]) > self.max_speech_frames:
            logger.warning(
                'Skipping overlength speech latent | index=%s | frames=%s | max_speech_frames=%s',
                row_index,
                int(latent.shape[0]),
                self.max_speech_frames
            )

            return None

        text = metadata_row[self.text_column]
        input_ids, attention_mask = self._tokenize(text)
        attention_mask = attention_mask.to(torch.bool)
        audio_path = None

        if self.need_audio_path:
            if audio_row is None:
                raise ValueError('audio_ds is required when streamed speaker audio is needed')

            audio_path = audio_row.get(self.audio_column, None)

        return (input_ids, attention_mask, latent, text, audio_path)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        latents_iter = iter(self.latents_ds)
        audio_iter = iter(self.audio_ds) if self.audio_ds is not None else None
        metadata_len = len(self.metadata_ds)

        for idx in range(metadata_len):
            try:
                latents_row = next(latents_iter)
            except StopIteration as exc:
                raise ValueError(f'latents_ds ended before metadata_ds at row {idx}; sources must be row-aligned') from exc

            audio_row = None

            if audio_iter is not None:
                try:
                    audio_row = next(audio_iter)
                except StopIteration as exc:
                    raise ValueError(f'audio_ds ended before metadata_ds at row {idx}; sources must be row-aligned') from exc

            if self.selected_indices is not None and idx not in self.selected_indices:
                continue

            if idx % num_workers != worker_id:
                continue

            item = self._row_to_item(self.metadata_ds[idx], latents_row, audio_row, row_index=idx)

            if item is not None:
                yield item

        try:
            next(latents_iter)
        except StopIteration:
            pass
        else:
            raise ValueError('latents_ds has more rows than metadata_ds; sources must be row-aligned')

        if audio_iter is not None:
            try:
                next(audio_iter)
            except StopIteration:
                pass
            else:
                raise ValueError('audio_ds has more rows than metadata_ds; sources must be row-aligned')


class KeyMatchedStreamingTTSDataset(MultiSourceStreamingTTSDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'key' not in self.metadata_ds.column_names:
            raise ValueError("key-matched streaming requires a 'key' column in metadata_ds")

        self.metadata_key_to_index = {}
        duplicate_rows = 0
        logger.info('Building streaming metadata key index | rows=%s', f'{len(self.metadata_ds):,}')

        offset = 0

        for batch in self.metadata_ds.select_columns(['key']).iter(batch_size = 100000):
            for idx, key in enumerate(batch['key'], start=offset):
                if key in self.metadata_key_to_index:
                    duplicate_rows += 1

                self.metadata_key_to_index[key] = idx

            offset += len(batch['key'])

            if offset % 1000000 == 0 or offset == len(self.metadata_ds):
                logger.info(
                    'Streaming metadata key index: %s/%s rows',
                    f'{offset:,}',
                    f'{len(self.metadata_ds):,}'
                )

        if duplicate_rows:
            logger.warning(
                'Streaming metadata contains %s duplicate-key rows; keeping the last row per key, consistent with indexed dataset key lookup',
                duplicate_rows
            )

    def __len__(self):
        raise TypeError('matched stream length is unknown until iteration completes')

    def __iter__(self):
        for latent_row in self.latents_ds:
            if 'key' not in latent_row:
                raise ValueError("key-matched streaming requires a 'key' column in latents_ds")

            idx = self.metadata_key_to_index.get(latent_row['key'])

            if idx is None or (self.selected_indices is not None and idx not in self.selected_indices):
                continue

            item = self._row_to_item(self.metadata_ds[idx], latent_row, None, row_index=idx)

            if item is not None:
                yield item


class IndexedDatasetStream(IterableDataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        workers = worker.num_workers if worker is not None else 1
        seed = torch.initial_seed() - worker_id
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(self.dataset), generator=generator)

        for idx in indices[worker_id::workers].tolist():
            yield self.dataset[idx]


class RandomInterleaveDataset(IterableDataset):

    def __init__(self, datasets, source_sizes):
        self.datasets = datasets
        self.source_sizes = source_sizes

    def __iter__(self):
        worker = get_worker_info()
        workers = worker.num_workers if worker else 1
        worker_id = worker.id if worker else 0
        rng = random.Random(torch.initial_seed())
        iterators = [iter(ds) for ds in self.datasets]
        remaining = [max(1, (size + workers - 1 - worker_id) // workers) for size in self.source_sizes]
        active = list(range(len(iterators)))

        while active:
            index = rng.choices(active, weights=[remaining[i] for i in active], k = 1)[0]

            try:
                item = next(iterators[index])
            except StopIteration:
                active.remove(index)
                continue

            remaining[index] = max(1, remaining[index] - 1)
            yield item


class StreamingTTSDataset(IterableDataset):

    def __init__(
        self,
        hf_dataset,
        latent_dim: int,
        need_audio_path: bool = False,
        tokenizer: Optional[Any] = None,
        max_text_len: Optional[int] = None,
        max_speech_frames: Optional[int] = None
    ):
        self.ds = hf_dataset
        self.latent_dim = latent_dim
        self.need_audio_path = need_audio_path
        self.tokenizer = tokenizer
        self.max_text_len = max_text_len
        self.max_speech_frames = None if max_speech_frames is None else int(max_speech_frames)

    def _tokenize(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError('requires a tokenizer for on-the-fly text.')

        enc = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt',
            add_special_tokens = True
        )

        return (enc['input_ids'][0], enc['attention_mask'][0])

    def _fix_latent_shape(self, latent: torch.Tensor) -> torch.Tensor:
        latent = latent.float()

        if latent.dim() == 1:
            latent = latent.unsqueeze(-1)

        if latent.shape[-1] != self.latent_dim and latent.shape[0] == self.latent_dim:
            latent = latent.T

        if latent.shape[-1] != self.latent_dim:
            raise ValueError(f'latent dim mismatch: expected {self.latent_dim}, got {tuple(latent.shape)}')

        return latent

    def _load_latent(self, row: dict) -> torch.Tensor:
        return self._fix_latent_shape(load_tensor(row['latents']))

    def _merged_keys(self, row: dict) -> Optional[list[str]]:
        merged = row.get('merged', '❌')

        if merged is None or merged == '❌':
            return None

        if not isinstance(merged, str):
            merged = str(merged)

        keys = [key for key in merged.split('➕') if key]

        return keys or None

    def _row_to_item(self, row: dict, key_to_latent: dict[str, torch.Tensor]):
        keys = self._merged_keys(row)

        if keys:
            latents = []
            missing = []

            for key in keys:
                if key in key_to_latent:
                    latents.append(key_to_latent[key])
                else:
                    missing.append(key)

            if missing:
                raise KeyError(f'merged key not found in streamed dataset cache: {missing}')

            latent = torch.concatenate(latents)
        else:
            latent = self._load_latent(row)

        if self.max_speech_frames is not None and int(latent.shape[0]) > self.max_speech_frames:
            logger.warning(
                'Skipping overlength speech latent | key=%s | frames=%s | max_speech_frames=%s',
                row.get('key', None),
                int(latent.shape[0]),
                self.max_speech_frames
            )

            return None

        text = row['text']
        input_ids, attention_mask = self._tokenize(text)
        attention_mask = attention_mask.to(torch.bool)
        audio_path = None

        if self.need_audio_path:
            audio_path = row.get('filename', row.get('audio', None))

        return (input_ids, attention_mask, latent, text, audio_path)

    def _can_emit(self, row: dict, key_to_latent: dict[str, torch.Tensor]) -> bool:
        keys = self._merged_keys(row)

        if not keys:
            return True

        return all((key in key_to_latent for key in keys))

    def __iter__(self):
        key_to_latent = {}
        pending = []

        for row in self.ds:
            key = row.get('key', None)

            if key is not None:
                key_to_latent[key] = self._load_latent(row)

            if self._can_emit(row, key_to_latent):
                item = self._row_to_item(row, key_to_latent)

                if item is not None:
                    yield item
            else:
                pending.append(row)

            if pending:
                still_pending = []

                for pending_row in pending:
                    if self._can_emit(pending_row, key_to_latent):
                        item = self._row_to_item(pending_row, key_to_latent)

                        if item is not None:
                            yield item
                    else:
                        still_pending.append(pending_row)

                pending = still_pending

        if pending:
            missing = {}

            for pending_row in pending:
                keys = self._merged_keys(pending_row) or []
                missing[pending_row.get('key', '<unknown>')] = [key for key in keys if key not in key_to_latent]

            raise KeyError(f'merged keys not found in streamed dataset cache: {missing}')


class TTSCollator:
    MIN_AUDIO_SAMPLES = 16000
    DEFAULT_BUCKETS = (125, 250, 376)

    def __init__(
        self,
        max_text_len: int,
        max_latent_len: int,
        text_pad_id: int,
        latent_dim: int,
        load_audio: bool = False,
        latent_length_buckets: Optional[list[int]]=None,
        latent_pad_value: float = 0.0,
        trim_text_padding: bool = False
    ):
        self.max_text_len = max_text_len
        self.max_latent_len = int(max_latent_len)
        self.text_pad_id = text_pad_id
        self.latent_dim = latent_dim
        self.load_audio = load_audio
        self.latent_pad_value = float(latent_pad_value)
        self.trim_text_padding = bool(trim_text_padding)
        buckets = list(self.DEFAULT_BUCKETS if latent_length_buckets is None else latent_length_buckets)
        buckets = sorted({int(b) for b in buckets if int(b) > 0})

        if not buckets:
            raise ValueError('latent_length_buckets must contain at least one positive bucket')

        if buckets[-1] != self.max_latent_len:
            raise ValueError(f'largest bucket must equal max_latent_len: buckets={buckets}, max_latent_len={self.max_latent_len}')

        self.latent_length_buckets = buckets

    def _select_bucket(self, speech_length: int) -> int:
        speech_length = int(speech_length)

        if speech_length <= 0:
            raise ValueError(f'speech latent must contain at least one frame, got {speech_length}')

        first = next((i for i, b in enumerate(self.latent_length_buckets) if b >= speech_length), None)

        if first is None:
            raise ValueError(f'speech is too long for available buckets without truncation: speech={speech_length}, buckets={self.latent_length_buckets}')

        eligible = self.latent_length_buckets[first:]

        if len(eligible) == 1:
            return eligible[0]

        r = float(torch.rand(()).item())

        if len(eligible) == 3:
            return eligible[0] if r < 0.6 else eligible[1] if r < 0.85 else eligible[2]

        if len(eligible) == 2:
            return eligible[0] if r < 0.75 else eligible[1]

        return eligible[int(torch.randint(0, len(eligible), (1,)).item())]

    def _load_audio_ref(self, audio_path):
        if audio_path is None:
            return None

        if isinstance(audio_path, (list, tuple)):
            parts = [self._load_audio_ref(part) for part in audio_path]

            if not parts or any((part is None for part in parts)):
                return None

            return np.concatenate(parts)

        if isinstance(audio_path, dict):
            if audio_path.get('array', None) is not None:
                wav = np.asarray(audio_path['array'], dtype=np.float32)
                sr = audio_path.get('sampling_rate', 16000)

                if sr != 16000:
                    wav = librosa.resample(wav, orig_sr=sr, target_sr = 16000)

                return wav

            if audio_path.get('bytes', None) is not None:
                wav, _ = librosa.load(io.BytesIO(audio_path['bytes']), sr = 16000)

                return wav

            audio_path = audio_path.get('path', None)

        if isinstance(audio_path, (bytes, bytearray)):
            wav, _ = librosa.load(io.BytesIO(audio_path), sr = 16000)

            return wav

        if isinstance(audio_path, os.PathLike):
            audio_path = os.fspath(audio_path)

        if isinstance(audio_path, str):
            wav, _ = librosa.load(audio_path, sr = 16000)

            return wav

        return None

    def __call__(self, batch):
        text_list, attn_list, latent_list, raw_texts, audio_paths = zip(*batch)
        batch_size = len(text_list)
        text_ids = torch.full((batch_size, self.max_text_len), self.text_pad_id, dtype=torch.long)
        text_mask = torch.zeros(batch_size, self.max_text_len, dtype=torch.bool)

        for i, (ids, attn) in enumerate(zip(text_list, attn_list)):
            L = min(len(ids), self.max_text_len)
            text_ids[i, :L] = ids[:L]
            text_mask[i, :L] = attn[:L]

        if self.trim_text_padding:
            length = int(text_mask.sum(dim = 1).max())
            length = min(self.max_text_len, max(512, math.ceil(length / 512) * 512))
            text_ids, text_mask = (text_ids[:, :length], text_mask[:, :length])

        speech_lengths = [int(lat.shape[0]) for lat in latent_list]
        sample_buckets = [self._select_bucket(length) for length in speech_lengths]
        batch_pad_len = max(sample_buckets)
        latents = torch.full(
            (batch_size, batch_pad_len, self.latent_dim),
            self.latent_pad_value,
            dtype=torch.float32
        )
        speech_mask = torch.zeros(batch_size, batch_pad_len, dtype=torch.bool)
        latent_mask = torch.zeros(batch_size, batch_pad_len, dtype=torch.bool)

        for i, (lat, speech_len, bucket_len) in enumerate(zip(latent_list, speech_lengths, sample_buckets)):
            if speech_len > bucket_len:
                raise AssertionError(f'bucket invariant violated: speech={speech_len}, bucket={bucket_len}')

            latents[i, :speech_len] = lat.float()
            speech_mask[i, :speech_len] = True
            latent_mask[i, :bucket_len] = True

        out = {
            'text_ids': text_ids,
            'text_mask': text_mask,
            'texts': list(raw_texts),
            'latents': latents,
            'latent_mask': latent_mask,
            'speech_mask': speech_mask,
            'speech_lengths': torch.tensor(speech_lengths, dtype=torch.long),
            'bucket_lengths': torch.tensor(sample_buckets, dtype=torch.long)
        }

        if self.load_audio:
            audio_arrays, audio_lengths, valid_flags = ([], [], [])

            for audio_path in audio_paths:
                if audio_path is not None:
                    try:
                        wav = self._load_audio_ref(audio_path)

                        if wav is not None and len(wav) >= self.MIN_AUDIO_SAMPLES:
                            audio_arrays.append(wav)
                            audio_lengths.append(len(wav))
                            valid_flags.append(True)
                            continue
                    except Exception:
                        pass

                audio_arrays.append(np.zeros(1, dtype=np.float32))
                audio_lengths.append(0)
                valid_flags.append(False)

            max_audio_len = max(audio_lengths) if any(valid_flags) else 1
            audio_batch = np.zeros((batch_size, max_audio_len), dtype=np.float32)

            for i, wav in enumerate(audio_arrays):
                if valid_flags[i]:
                    audio_batch[i, :len(wav)] = wav

            out['speaker_audio'] = torch.from_numpy(audio_batch)
            out['speaker_audio_lengths'] = torch.tensor(audio_lengths, dtype=torch.long)
            out['speaker_audio_valid'] = torch.tensor(valid_flags, dtype=torch.bool)

        return out
_SINGLE_FILE_BUILDERS = {'.parquet': 'parquet', '.arrow': 'arrow'}


def _single_file_builder(dataset_path: str) -> Optional[str]:
    ext = os.path.splitext(dataset_path)[1].lower()
    builder = _SINGLE_FILE_BUILDERS.get(ext)

    if builder is None:
        return None

    if dataset_path.startswith('s3://') or os.path.isfile(dataset_path):
        return builder

    return None


def _load_streaming_dataset(dataset_path: str, dataset_split: str, train_cfg: dict, columns = None):
    storage_options = {
        'default_block_size': train_cfg.get('s3_block_size', 32 * 1024 * 1024),
        'default_cache_type': 'readahead'
    }

    for k_env, k_opt in [
        ('AWS_ACCESS_KEY_ID', 'key'),
        ('AWS_SECRET_ACCESS_KEY', 'secret'),
        ('AWS_SESSION_TOKEN', 'token'),
        ('AWS_DEFAULT_REGION', 'client_kwargs')
    ]:
        v = os.environ.get(k_env)

        if not v:
            continue

        if k_opt == 'client_kwargs':
            storage_options['client_kwargs'] = {'region_name': v}
        else:
            storage_options[k_opt] = v

    is_s3 = dataset_path.startswith('s3://')
    is_local_dir = os.path.isdir(dataset_path)
    single_builder = _single_file_builder(dataset_path)

    if single_builder is not None:
        return load_dataset(
            single_builder,
            **{'columns': columns} if columns is not None and single_builder == 'parquet' else {},
            data_files={dataset_split: [dataset_path]},
            split=dataset_split,
            streaming = True,
            storage_options=storage_options if is_s3 else None
        )

    if not (is_s3 or is_local_dir):
        return load_dataset(
            dataset_path,
            split=dataset_split,
            streaming = True,
            **{'columns': columns} if columns is not None else {}
        )

    if is_s3:
        fs = s3fs.S3FileSystem(**{k: v for k, v in storage_options.items() if k in ('key', 'secret', 'token', 'client_kwargs')})
        base = dataset_path.rstrip('/')
        parquet_files = sorted(('s3://' + p for p in fs.glob(f'{base[5:]}/**/*.parquet')))
        arrow_files = sorted(('s3://' + p for p in fs.glob(f'{base[5:]}/**/*.arrow')))
    else:
        parquet_files = sorted(glob(os.path.join(dataset_path, '**', '*.parquet'), recursive = True))
        arrow_files = sorted(glob(os.path.join(dataset_path, '**', '*.arrow'), recursive = True))

    if parquet_files:
        builder, files = ('parquet', parquet_files)
    elif arrow_files:
        builder, files = ('arrow', arrow_files)
    else:
        raise FileNotFoundError(f'No parquet or arrow shards found under {dataset_path}')

    return load_dataset(
        builder,
        **{'columns': columns} if columns is not None and builder == 'parquet' else {},
        data_files={dataset_split: files},
        split=dataset_split,
        streaming = True,
        storage_options=storage_options if is_s3 else None
    )


def _load_nonstreaming_dataset(dataset_path: str):
    single_builder = _single_file_builder(dataset_path)

    if single_builder is not None:
        logger.info(f'Dataset load attempt: {dataset_path} | mode=single_file({single_builder})')

        ds = load_dataset(single_builder, data_files=dataset_path, split = 'train')
        logger.info(f'Dataset load success: {dataset_path} | mode=single_file({single_builder}) | rows={len(ds):,}')

        return ds

    logger.info(f'Dataset load attempt: {dataset_path} | mode=load_from_disk')

    try:
        ds = get_first_split(load_from_disk(dataset_path))
        logger.info(f'Dataset load success: {dataset_path} | mode=load_from_disk | rows={len(ds):,}')

        return ds
    except Exception as exc:
        logger.warning(f'Dataset load_from_disk failed: {dataset_path} | {type(exc).__name__}: {exc}')

    logger.info(f'Dataset load fallback: {dataset_path} | mode=load_dataset')

    ds = get_first_split(load_dataset(dataset_path))
    logger.info(f'Dataset load success: {dataset_path} | mode=load_dataset | rows={len(ds):,}')

    return ds


def build_multi_tts_dataloader(
    dataset_configs: list[dict],
    latent_dim: int,
    need_audio: bool,
    max_text_len: int,
    max_latent_len: int,
    text_pad_id: int,
    train_cfg: dict,
    tokenizer: Optional[Any]=None,
    streaming: bool = False,
    dataset_split: str = 'train',
    buckets: Optional[list[int]]=None,
    pad_value: float = 0.0
):
    length_stage = bool(train_cfg.get('length_training_stage', False))
    has_streaming = any((bool(cfg.get('streaming', streaming)) for cfg in dataset_configs))

    if length_stage and has_streaming:
        raise ValueError('length training requires indexed, non-streaming datasets')

    tokenizer = _load_tokenizer(tokenizer)
    max_speech_frames = int(max_latent_len)
    datasets = []

    for dataset_cfg in dataset_configs:
        metadata_path = dataset_cfg['metadata_ds']
        latents_path = dataset_cfg['latents_ds']
        audio_path = dataset_cfg.get('audio_ds')
        use_percentage = float(dataset_cfg.get('use_percentage', 1.0))

        if not 0.0 < use_percentage <= 1.0:
            raise ValueError(f'use_percentage must be in (0, 1], got {use_percentage}')

        must_merge = bool(dataset_cfg.get('must_merge', False))

        if length_stage and must_merge:
            raise ValueError('length training expects individual rows; set must_merge=false')

        audio_column = dataset_cfg.get('audio_column', 'audio')
        latents_column = dataset_cfg.get('latents_columns', dataset_cfg.get('latents_column', 'latents'))
        text_column = dataset_cfg.get('text_column', 'text')
        metadata_ds = _load_nonstreaming_dataset(metadata_path)
        source_streaming = bool(dataset_cfg.get('streaming', streaming))

        if source_streaming:
            if need_audio:
                raise ValueError('per-source streaming currently supports base training without speaker audio')

            if must_merge:
                raise ValueError('must_merge=true is not supported with --streaming because streamed latents require row-aligned access')

            latents_ds = _load_streaming_dataset(
                latents_path,
                dataset_split=dataset_split,
                train_cfg=train_cfg,
                columns=['key', latents_column]
            )
            audio_ds = None

            if need_audio:
                if not audio_path:
                    raise ValueError('audio_ds is required when --streaming is used with speaker audio')

                audio_ds = _load_streaming_dataset(
                    audio_path,
                    dataset_split=dataset_split,
                    train_cfg=train_cfg
                )

                try:
                    audio_ds = audio_ds.cast_column(audio_column, Audio(sampling_rate = 16000))
                except Exception:
                    pass

            selected_indices = None

            if use_percentage < 1.0:
                seed = int(train_cfg.get('seed', 42))
                count = max(1, int(len(metadata_ds) * use_percentage))
                indices = list(range(len(metadata_ds)))
                random.Random(seed).shuffle(indices)

                selected_indices = set(indices[:count])

            datasets.append(KeyMatchedStreamingTTSDataset(metadata_dataset=metadata_ds, latents_dataset=latents_ds, latent_dim=latent_dim, need_audio_path=need_audio, tokenizer=tokenizer, max_text_len=max_text_len, audio_dataset=audio_ds, audio_column=audio_column, latents_column=latents_column, text_column=text_column, selected_indices=selected_indices, max_speech_frames=max_speech_frames))
            continue

        latents_ds = metadata_ds if latents_path == metadata_path else _load_nonstreaming_dataset(latents_path)
        audio_ds = _load_nonstreaming_dataset(audio_path) if need_audio and audio_path else None

        if use_percentage < 1.0 and (not length_stage):
            seed = int(train_cfg.get('seed', 42))
            count = max(1, int(len(metadata_ds) * use_percentage))
            indices = list(range(len(metadata_ds)))
            random.Random(seed).shuffle(indices)

            indices = indices[:count]
            metadata_ds = metadata_ds.select(indices)

            if latents_path == metadata_path:
                latents_ds = metadata_ds

            if audio_path == metadata_path:
                audio_ds = metadata_ds

        dataset_type = LengthTrainingDataset if length_stage else TTSDataset
        length_kwargs = dict(
            codec_rate_hz=train_cfg['length_codec_rate_hz'],
            max_concat_samples = 8,
            max_concat_seconds = 60.0,
            selection_fraction=train_cfg.get('length_training_fraction', 0.5) * use_percentage,
            seed=train_cfg.get('seed', 42),
            planning_workers=train_cfg.get('length_planning_workers', 32),
            silence_latent_path=train_cfg.get('length_silence_latent_path')
        ) if length_stage else {}
        source_dataset = dataset_type(
            metadata_ds,
            latent_dim=latent_dim,
            need_audio_path=need_audio,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            latents_dataset=latents_ds,
            audio_dataset=audio_ds,
            must_merge=must_merge,
            audio_column=audio_column,
            latents_column=latents_column,
            text_column=text_column,
            max_speech_frames=max_speech_frames,
            key_index_cache_dir=train_cfg.get(
                'length_index_cache_dir',
                train_cfg.get('key_index_cache_dir')
            ),
            **length_kwargs
        )

        if length_stage and (not len(source_dataset)):
            logger.warning('No eligible length-training groups in %s; source omitted', metadata_path)
            continue

        datasets.append(source_dataset)

    if not datasets:
        raise ValueError('No eligible length-training groups were found' if length_stage else 'datasets config must contain at least one dataset')

    if has_streaming:
        source_sizes = [(len(ds.selected_indices) if ds.selected_indices is not None else len(ds.metadata_ds)) if isinstance(ds, KeyMatchedStreamingTTSDataset) else len(ds) for ds in datasets]
        datasets = [ds if isinstance(ds, IterableDataset) else IndexedDatasetStream(ds) for ds in datasets]
        dataset = datasets[0] if len(datasets) == 1 else RandomInterleaveDataset(datasets, source_sizes)
        logger.info('Randomly interleaving training sources | nominal rows=%s', source_sizes)
    else:
        dataset = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    collator = TTSCollator(
        max_text_len=max_text_len,
        max_latent_len=max_latent_len,
        text_pad_id=text_pad_id,
        latent_dim=latent_dim,
        load_audio=need_audio,
        latent_length_buckets=buckets or train_cfg.get('latent_length_buckets'),
        latent_pad_value=pad_value,
        trim_text_padding=length_stage
    )
    num_workers = int(train_cfg.get('streaming_num_workers', 0)) if has_streaming else train_cfg.get('num_workers', 12)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle = False if has_streaming else True,
        num_workers=num_workers,
        pin_memory = True,
        drop_last = True,
        collate_fn=collator,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    )

    return (dataset, dataloader)


def build_tts_dataloader(
    dataset_path: str,
    latent_dim: int,
    need_audio: bool,
    max_text_len: int,
    max_latent_len: int,
    text_pad_id: int,
    train_cfg: dict,
    tokenizer: Optional[Any]=None,
    streaming: bool = False,
    dataset_split: str = 'train',
    buckets: Optional[list[int]]=None,
    pad_value: float = 0.0
):
    length_stage = bool(train_cfg.get('length_training_stage', False))

    if length_stage and streaming:
        raise ValueError('length training requires indexed, non-streaming datasets')

    tokenizer = _load_tokenizer(tokenizer)
    max_speech_frames = int(max_latent_len)

    if streaming:
        hf_ds = _load_streaming_dataset(dataset_path, dataset_split=dataset_split, train_cfg=train_cfg)

        if need_audio:
            try:
                column_names = hf_ds.column_names
            except Exception:
                column_names = None

            if column_names is None or 'filename' in (column_names or []):
                try:
                    hf_ds = hf_ds.cast_column('filename', Audio(sampling_rate = 16000))
                except Exception:
                    pass

        shuffle_buffer = train_cfg.get('streaming_shuffle_buffer', 10000)

        if shuffle_buffer:
            hf_ds = hf_ds.shuffle(buffer_size=shuffle_buffer, seed=train_cfg.get('seed', 42))

        dataset = StreamingTTSDataset(
            hf_ds,
            latent_dim=latent_dim,
            need_audio_path=need_audio,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_speech_frames=max_speech_frames
        )
    else:
        if length_stage:
            hf_ds = _load_nonstreaming_dataset(dataset_path)
        else:
            try:
                hf_ds = get_first_split(load_from_disk(dataset_path))
            except:
                hf_ds = get_first_split(load_dataset(dataset_path))

        dataset_type = LengthTrainingDataset if length_stage else TTSDataset
        length_kwargs = dict(
            codec_rate_hz=train_cfg['length_codec_rate_hz'],
            max_concat_samples = 8,
            max_concat_seconds = 60.0,
            selection_fraction=train_cfg.get('length_training_fraction', 0.5),
            seed=train_cfg.get('seed', 42),
            planning_workers=train_cfg.get('length_planning_workers', 32),
            silence_latent_path=train_cfg.get('length_silence_latent_path')
        ) if length_stage else {}
        dataset = dataset_type(
            hf_ds,
            latent_dim=latent_dim,
            need_audio_path=need_audio,
            tokenizer=tokenizer,
            max_text_len=max_text_len,
            max_speech_frames=max_speech_frames,
            key_index_cache_dir=train_cfg.get(
                'length_index_cache_dir',
                train_cfg.get('key_index_cache_dir')
            ),
            **length_kwargs
        )

        if length_stage and (not len(dataset)):
            raise ValueError('No eligible length-training groups were found')

    collator = TTSCollator(
        max_text_len=max_text_len,
        max_latent_len=max_latent_len,
        text_pad_id=text_pad_id,
        latent_dim=latent_dim,
        load_audio=need_audio,
        latent_length_buckets=buckets or train_cfg.get('latent_length_buckets'),
        latent_pad_value=pad_value,
        trim_text_padding=length_stage
    )
    num_workers = train_cfg.get('num_workers', 4 if streaming else 12)
    dataloader = DataLoader(
        dataset,
        batch_size=train_cfg['batch_size'],
        shuffle = False if streaming else True,
        num_workers=num_workers,
        pin_memory = True,
        drop_last = True,
        collate_fn=collator,
        persistent_workers=num_workers > 0,
        prefetch_factor=train_cfg.get('prefetch_factor', 4) if num_workers > 0 else None
    )

    return (dataset, dataloader)
