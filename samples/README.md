# Sample audio

Audio files are **not** stored in the repository (they are gitignored).

1. Add any short **16 kHz mono WAV** here (e.g. `my_clip.wav`), or use your own path.
2. Convert to PCM:

   ```powershell
   python scripts/wav_to_pcm.py samples/my_clip.wav samples/my_clip.pcm
   ```

3. Stream it to a running server:

   ```powershell
   python scripts/send_test_pcm.py samples/my_clip.pcm
   ```
