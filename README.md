This video format is a Lightweight lossless-video format with a custom container and VP8L as the base underlying codec.

The .miku format is designed to be able to be played back on old hardware at reasonable frame rate
(playback speed optimization is still WIP although it is currently quite good for a lossless video format. 

This format uses FLAC as the audio codec so that the audio is lossless as well. 

Features so far: 
Frame deduplication.
Delta frames.
Scrubbing support.
VFR support.

Features planned to add:
Embedded .ASS subtitle support.
Chapters.
Multiple episodes within one file.
Custom compression algorithm (type of dedup)
Thing that remembers where the video was when you last played it so that you can resume.

Features that are WIP:
Multi track flac audio (for multi language videos or ones with directors commentary)
Thumbnails.
Metadata.

Remember this is still in development so it still has a lot of work to be done and many more features to be implemented.

Encoder = mikuenc.c
Muxer = mikumux.c 
Demuxer = mikudec.c
Decoder = mikudeltadec.c

To give you an idea of the preformance of this format i was able to decode 720p at 24fps smoothly on a Intel Core 2 Duo P8400 (with 2GB ram) with room to spare.

Install instructions: 

Open the location of build-miku-ffmpeg.sh in your terminal and run it (sh builld-miku-ffmpeg) Be carefull it did work fine on my linux machines (EndeavorOS)
and it should work on most distros but it is still a script. 

This project is Vibe-coded using claude (sonnet 4.6 to 5.5 MAX effort with extended thinking) but all ideas and innovations are mine and NOT claudes. Development started around May of 2026. 


