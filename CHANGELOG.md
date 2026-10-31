# Changelog

## 1.0.4

- **Added:** a sigma for each graph. **Suggest limits** now uses a separate sigma for FWHM, eccentricity, star count, background and SNR, set the same way round as PixInsight's SubframeSelector (plus for too high, minus for too low). The starting values are FWHM 2, eccentricity 2, background 2.5, star count -1.5 and SNR -2.5. Changing a sigma moves its slider straight away, and "(min. gap)" shows when the minimum gap, not the sigma, sets a limit.
- **Added:** Sort into folders can put the folders **next to** each file's folder (`Lights_Bad` beside `Lights`) instead of inside it, so stacking software that reads the folders inside your lights folder does not pick up the bad frames.
- **Added:** **Only move the Bad frames** in Sort into folders: the bad frames go, everything else stays where it is.
- **Fixed:** "and the rest Good" only worked when ticked before pressing **Mark them Bad**; afterwards it did nothing. It is now its own **Mark the rest Good** button, which works at any time.
- **Fixed:** Sort into folders gave the wrong counts when SER frames were sorted ("Move 0 files", and after a failed SER write even "Moved -1 files"). It now says how many files it moves and how many SER files it writes.
- **Fixed:** sorting or undoing while the blink was playing could, on Windows, catch a sub while it was being read, leaving a copy in both folders. Playback now stops and any frame being read finishes first.
- **Changed:** a tidier window. The blink controls are in a bar under the picture with Show trails, Magnifier, Fit and 1:1. Theme, Debayer, Lock layout and Reset layout are in a new **Settings** window, and the log is under the arrow beside **Help**. One toolbar row instead of two.

## 1.0.3

- **Added:** an update check. Once a day AS Blink asks GitHub whether a newer version has been released. If there is one, a bar appears under the toolbar with an **Open the download page** button. Only the version number is asked for: nothing about you or your images is sent.
- The Help window has a **Check GitHub for new versions (once a day)** tick box to turn it off, and a **Check now** button.

## 1.0.2

- **Fixed:** on Windows, a file sorted into the Good or Bad folder could be saved under its short 8.3 name (such as `LIGH~12D.FIT`) instead of its full name. This happened with very long paths, for example long N.I.N.A. file names in deep folders on a shared drive. AS Blink now always uses the full file name and handles paths over Windows' 260-character limit.
- **Added:** a **Focuser temp** graph, read from FOCTEMP (or FOCUSTEM) in the header. It shows whether a change in FWHM follows the temperature or the seeing.
- The **Temperature** graph is now called **Sensor temp**, so the two are not mixed up.
- A graph for a header value that the files don't have now says "not in these files' headers" instead of "no data yet".
- **Added:** opening a SER or video longer than 2,000 frames asks whether to load all of it or just a number of frames from a chosen start frame. Only those frames are measured, so a long capture is ready in seconds, and Sort into folders can save them as a new, shorter SER.
- **Fixed:** planetary and lunar SER or video captures with a near-black sky were stretched so hard the planet blew out to white. The **Linear** stretch now takes its white point from the planet or Moon itself, and SER and video files open on Linear automatically. Your own stretch choice comes back when you open subs again.

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
