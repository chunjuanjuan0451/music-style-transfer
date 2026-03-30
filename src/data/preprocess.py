from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pretty_midi

SEQ_LEN: int = 32          
FS: int = 12               
PITCH_LOW: int = 21        
PITCH_HIGH: int = 109    
INPUT_DIM: int = 89        

TEMPO_MIN: float = 40.0
TEMPO_MAX: float = 200.0
DURATION_MIN: float = 10.0
DURATION_MAX: float = 300.0
MIN_NOTES: int = 50       

STRIDE: int = 16           
NOTE_DENSITY_MIN: float = 0.02
NOTE_DENSITY_MAX: float = 0.15

SPLIT_RATIOS: Tuple[float, float, float] = (0.80, 0.10, 0.10)

GENRE_KEYWORDS: Dict[str, List[str]] = {
    "classical": [],        
    "jazz": [
        "jazz",
        "blues",
        "rnb",             
    ],
    "pop": [
        "pop",
        "rock",
        "electronic",
        "rap",
        "metal",
        "country",
        "reggae",
        "latin",
        "folk",
        "punk",
        "world",
        "new age",
    ],
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)



def load_genre_map(metadata_path: str) -> Dict[str, str]:
    ext = Path(metadata_path).suffix.lower()
    if ext == ".h5":
        return _load_genre_map_h5(metadata_path)
    else:
        return _load_genre_map_cls(metadata_path)


def _load_genre_map_h5(metadata_path: str) -> Dict[str, str]:
    try:
        import h5py
    except ImportError:
        raise ImportError(
            "h5py is required to read .h5 metadata.\n"
            "Install: pip install h5py"
        )

    genre_map: Dict[str, str] = {}
    with h5py.File(metadata_path, "r") as f:
        if "metadata" in f and "songs" in f["metadata"]:
            # Standard MSD summary file layout
            songs     = f["metadata"]["songs"]
            track_ids = songs["track_id"][:]
            genres    = songs["genre"][:]
            for tid, g in zip(track_ids, genres):
                tid_str = tid.decode("utf-8") if isinstance(tid, bytes) else str(tid)
                g_str   = g.decode("utf-8")   if isinstance(g,   bytes) else str(g)
                if tid_str and g_str:
                    genre_map[tid_str.strip()] = g_str.strip().lower()
        else:
            for key in f.keys():
                try:
                    grp = f[key]
                    genre = None
                    if "genre" in grp.attrs:
                        genre = grp.attrs["genre"]
                    elif "metadata" in grp and "genre" in grp["metadata"].attrs:
                        genre = grp["metadata"].attrs["genre"]
                    if genre is not None:
                        g_str = genre.decode("utf-8") if isinstance(genre, bytes) else str(genre)
                        genre_map[key.strip()] = g_str.strip().lower()
                except Exception:
                    continue

    if not genre_map:
        raise ValueError(
            f"No genre labels found in {metadata_path}.\n"
            "Inspect structure: python -c \"import h5py; "
            "f=h5py.File('lmd_matched_metadata.h5','r'); print(list(f.keys()))\""
        )

    log.info("Loaded %d genre labels from .h5 metadata.", len(genre_map))
    return genre_map


def _load_genre_map_cls(metadata_path: str) -> Dict[str, str]:
    genre_map: Dict[str, str] = {}
    with open(metadata_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                msd_id = parts[0]
                genre  = parts[1].lower()   
                genre_map[msd_id] = genre
    log.info("Loaded %d genre labels from .cls metadata.", len(genre_map))
    return genre_map


def resolve_target_genre(raw_genre: str, target_genres: List[str]) -> Optional[str]:
    raw_lower = raw_genre.lower().strip()
    for tgt in target_genres:
        if raw_lower in GENRE_KEYWORDS.get(tgt, []):
            return tgt
    return None


def passes_quality_filter(pm: pretty_midi.PrettyMIDI) -> bool:
    duration = pm.get_end_time()
    if not (DURATION_MIN <= duration <= DURATION_MAX):
        return False

    try:
        tempo_change_times = pm.get_tempo_change_times()   
        resolution = pm.resolution  
        tempos = np.array([
            60.0 / (spt * resolution)
            for _, spt in pm._tick_scales
        ])
        if len(tempos) == 0:
            mean_tempo = 120.0
        else:
            times = np.concatenate([tempo_change_times, [duration]])
            durations = np.diff(times)
            n = min(len(tempos), len(durations))
            if n == 0 or durations[:n].sum() < 1e-6:
                mean_tempo = float(tempos[0]) if len(tempos) > 0 else 120.0
            else:
                mean_tempo = float(
                    np.average(tempos[:n], weights=durations[:n])
                )
    except Exception:
        mean_tempo = 120.0

    if not (TEMPO_MIN <= mean_tempo <= TEMPO_MAX):
        return False

    non_drum_notes = sum(
        len(inst.notes)
        for inst in pm.instruments
        if not inst.is_drum
    )
    return non_drum_notes >= MIN_NOTES

def extract_piano_roll(pm: pretty_midi.PrettyMIDI) -> np.ndarray:
    roll = pm.get_piano_roll(fs=FS) 
    roll = roll[PITCH_LOW:PITCH_HIGH, :]  
    roll = (roll > 0).astype(np.float32)
    return roll


def segment_piano_roll(roll: np.ndarray) -> np.ndarray:
    num_pitches, T = roll.shape
    segments: List[np.ndarray] = []

    for start in range(0, T - SEQ_LEN + 1, STRIDE):
        seg = roll[:, start : start + SEQ_LEN]  
        rest_token = (seg.sum(axis=0) == 0).astype(np.float32)
        seg_t = seg.T 
        seg_full = np.concatenate(
            [seg_t, rest_token[:, np.newaxis]], axis=1
        )  

        # Sparsity filter
        note_density = float(seg_t.mean())
        if NOTE_DENSITY_MIN <= note_density <= NOTE_DENSITY_MAX:
            segments.append(seg_full)

    if not segments:
        return np.empty((0, SEQ_LEN, INPUT_DIM), dtype=np.float32)
    return np.stack(segments, axis=0).astype(np.float32)


def discover_midi_files(
    midi_root: str,
    genre_map: Dict[str, str],
    target_genres: List[str],
    max_per_genre: int = 1000,
) -> Dict[str, List[str]]:
    midi_root_path = Path(midi_root)
    genre_files: Dict[str, List[str]] = {g: [] for g in target_genres}

    seen_msd_ids: set = set()

    for midi_path in sorted(midi_root_path.rglob("*.mid")):
        msd_id = midi_path.parent.name

        if msd_id in seen_msd_ids:
            continue

        raw_genre = genre_map.get(msd_id)
        if raw_genre is None:
            continue

        tgt = resolve_target_genre(raw_genre, target_genres)
        if tgt is None:
            continue

        if len(genre_files[tgt]) < max_per_genre:
            genre_files[tgt].append(str(midi_path))
            seen_msd_ids.add(msd_id)

    for genre, files in genre_files.items():
        log.info("Genre %-12s — %d files found.", genre, len(files))
    return genre_files


def split_segments(
    segments: np.ndarray,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(segments))
    segments = segments[idx]

    n = len(segments)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val   = int(n * SPLIT_RATIOS[1])

    train = segments[:n_train]
    val   = segments[n_train : n_train + n_val]
    test  = segments[n_train + n_val :]
    return train, val, test


def process_genre(
    midi_files: List[str],
    genre_name: str,
    out_dir: str,
    seed: int = 42,
) -> None:
    random.seed(seed)
    all_segments: List[np.ndarray] = []

    for i, midi_path in enumerate(midi_files):
        if (i + 1) % 50 == 0:
            log.info("  [%s] %d / %d files processed …", genre_name, i + 1, len(midi_files))
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception as exc:
            log.debug("  Skip %s — load error: %s", midi_path, exc)
            continue

        if not passes_quality_filter(pm):
            log.debug("  Skip %s — quality filter.", midi_path)
            continue

        roll = extract_piano_roll(pm)
        segs = segment_piano_roll(roll)
        if len(segs) > 0:
            all_segments.append(segs)

    if not all_segments:
        log.warning("[%s] No valid segments found — skipping.", genre_name)
        return

    all_segs = np.concatenate(all_segments, axis=0)
    log.info("[%s] Total segments before split: %d", genre_name, len(all_segs))

    train, val, test = split_segments(all_segs, seed=seed)
    log.info(
        "[%s] Split → train=%d  val=%d  test=%d",
        genre_name, len(train), len(val), len(test),
    )

    genre_out = Path(out_dir) / genre_name
    genre_out.mkdir(parents=True, exist_ok=True)

    np.save(str(genre_out / "train.npy"), train)
    np.save(str(genre_out / "val.npy"),   val)
    np.save(str(genre_out / "test.npy"),  test)
    log.info("[%s] Saved to %s", genre_name, genre_out)

# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LMD MIDI files → float32 piano roll .npy arrays."
    )
    parser.add_argument(
        "--midi_root",
        required=True,
        help="Root directory of the lmd_matched dataset.",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help=(
            "Path to genre metadata file. Accepts:\n"
            "  (1) lmd_matched_metadata.h5  — preferred, ships with lmd_matched\n"
            "                                  requires: pip install h5py\n"
            "  (2) msd_tagtraum_cd2.cls     — fallback tab-separated text file"
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for processed .npy files.",
    )
    parser.add_argument(
        "--genres",
        nargs="+",
        default=["classical", "jazz", "pop"],
        help="Target genres to process (default: classical jazz pop).",
    )
    parser.add_argument(
        "--max_per_genre",
        type=int,
        default=1000,
        help="Max MIDI files to load per genre (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log.info("Music Style Transfer — Data Preprocessing")
    log.info("MIDI root : %s", args.midi_root)
    log.info("Metadata  : %s", args.metadata)
    log.info("Output dir: %s", args.out_dir)
    log.info("Genres    : %s", args.genres)
    log.info(
        "lmd_matched path structure: "
        "<root>/<A>/<B>/<C>/<MSD_TRACK_ID>/<hash>.mid"
    )

    genre_map = load_genre_map(args.metadata)
    log.info("Genre map size: %d track IDs", len(genre_map))
    genre_files = discover_midi_files(
        args.midi_root, genre_map, args.genres, args.max_per_genre
    )

    for genre, files in genre_files.items():
        if not files:
            log.warning("No files found for genre '%s' — skipping.", genre)
            continue
        log.info("Processing genre: %s (%d files) …", genre, len(files))
        process_genre(files, genre, args.out_dir, seed=args.seed)

    log.info("Preprocessing complete.")
    log.info(
        "Upload the '%s' directory to Google Drive → "
        "MyDrive/music-style-transfer/data/processed/",
        args.out_dir,
    )


if __name__ == "__main__":
    main()