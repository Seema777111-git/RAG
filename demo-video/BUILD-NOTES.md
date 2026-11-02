# How this film is built

- `assets/live-session.mp4`: a real 148 s recording of the running app (localhost:8501), captured with Chrome's screencast
  at 30 fps with a drawn cursor. `assets/live-session-marks.json` has the time of each step.
- `index.html` cuts it into 11 clips (`data-media-start` + `data-playback-rate`). Waiting time is sped up and labelled on
  screen with its real duration; everything else plays at about real speed.
- `compositions/live-overlay.html`: title card, step labels 1-7, speed badges.
- `compositions/s07.html`: the pipeline diagram. `compositions/s09.html`: tests and closing line.
- `compositions/captions.html`: one caption per spoken sentence.
- Voice: `assets/voice/*.wav` (Windows Zira). No music bed (no music source without a HeyGen sign-in).
