#!/usr/bin/env bash
# download_essentia_musicnn_models.sh

mkdir -p essentia_models
cd essentia_models

base="https://essentia.upf.edu/models/classifiers"

files=(
  "mood_happy/mood_happy-musicnn-msd-2.pb"
  "mood_sad/mood_sad-musicnn-msd-2.pb"
  "mood_relaxed/mood_relaxed-musicnn-msd-2.pb"
  "mood_acoustic/mood_acoustic-musicnn-msd-2.pb"
  "mood_aggressive/mood_aggressive-musicnn-msd-2.pb"
  "mood_electronic/mood_electronic-musicnn-msd-2.pb"
  "danceability/danceability-musicnn-msd-2.pb"
  "gender/gender-musicnn-msd-2.pb"
)

for file in "${files[@]}"; do
  url="$base/$file"
  echo "⬇️  Downloading $file …"
  curl -L -O "$url"
done

echo "✅  All models are in $(pwd)"