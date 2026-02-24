import os
import json
import librosa
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, freeze_support

# ================= CONFIG =================

DATASET_ROOT = r"Y:\Dataset\actual_dataset"
OUTPUT_DIR = r"Y:\Dataset\audio_features"

PROCESSED_LOG = os.path.join(OUTPUT_DIR, "processed.json")
FEATURES_FILE = os.path.join(OUTPUT_DIR, "features.npy")
PATHS_FILE = os.path.join(OUTPUT_DIR, "paths.npy")

SR = 22050
N_MFCC = 40
WORKERS = max(1, cpu_count() - 2)

AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac")

# =========================================

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SR, mono=True)

        features = np.concatenate((
            librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC).mean(axis=1),
            librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
            librosa.feature.spectral_centroid(y=y, sr=sr).mean(axis=1),
            librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(axis=1),
            librosa.feature.zero_crossing_rate(y).mean(axis=1)
        ))

        return file_path, features

    except Exception:
        return ()

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== LOAD RESUME STATE ==========
    try:
        with open(PROCESSED_LOG, "r") as f:
            processed = set(json.load(f))
    except Exception:
        processed = set()

    # ========== DISCOVER AUDIO FILES ==========
    all_audio_files = {
        os.path.join(root, file)
        for root, _, files in os.walk(DATASET_ROOT)
        for file in files
        if file.lower().endswith(AUDIO_EXTENSIONS)
    }

    audio_files = list(all_audio_files - processed)

    print(f"New files to process: {len(audio_files)}")

    features_acc = []
    paths_acc = []

    # ========== MULTIPROCESSING ==========
    try:
        with Pool(WORKERS) as pool:
            for result in tqdm(
                pool.imap_unordered(extract_features, audio_files),
                total=len(audio_files),
                desc="Extracting audio features",
                unit="file"
            ):
                try:
                    path, feat = result
                    features_acc.append(feat)
                    paths_acc.append(path)
                    processed.add(path)
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n⏸ Extraction paused by user")

    # ========== SAVE STATE ==========
    try:
        with open(PROCESSED_LOG, "w") as f:
            json.dump(list(processed), f)

        np.save(FEATURES_FILE, np.array(features_acc))
        np.save(PATHS_FILE, np.array(paths_acc))

        print("State saved successfully")

    except Exception as e:
        print(f"Error while saving state: {e}")

# ========== WINDOWS ENTRY POINT ==========
if __name__ == "__main__":
    freeze_support()
    main()
