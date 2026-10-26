# Changelog

## 1.0.1

- **Videos:** opens AVI, MP4 and MOV files as well as SER. Every frame is a row in the table: step through them one at a time with Left / Right, or play at 0.5 to 30 frames a second.
- **Quick loading for SER and video:** frames are read for viewing and brightness only, so a long video is ready to play in seconds. The new option **Measure stars in SER and video frames (slow)** in More options brings back full star measuring.
- The "load them all?" question now only appears above 20,000 frames (was 3,000).
- Smoother playback and faster loading with thousands of frames.
- Colour or mono video is detected automatically, including compressed AVIs.
- **Sort into folders** leaves frames inside AVI, MP4 and MOV files out, and says so.
- Graphs that cannot be filled for video frames now say "not measured for video frames".
- The "Keep best" label in More options is no longer cut off.
- Wording: "Also write 2 new SER files" is now counted correctly.
- **Freeware licence** added to the code, the Help page and LICENCE.txt.

## 1.0.0

First release for beta testing.

- Blink comparator for FITS, XISF, TIFF, PNG, JPG, BMP, WebP, camera RAW and SER files.
- Correct OSC debayering, with the Bayer pattern checked against the data.
- Statistics per frame, session graphs with limit sliders, Suggest limits and Keep only the best.
- Trail detection, alignment to a reference frame, auto stretch, full header view, magnifier.
- Mark Good / Bad, sort into folders with undo, GIF / MP4 and CSV export.
- Dark, Light and Night (red) themes; the layout is remembered and can be locked.
- Support log: **Log > Save log for support...**
