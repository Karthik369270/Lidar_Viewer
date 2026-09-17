# LiDAR / camera viewer

Shows a camera frame and the matching LiDAR revolution side by side, with
motion deskewing switchable on and off.

## Run

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and drop in:

| File | Required | Notes |
|---|---|---|
| LiDAR `.pcap` | yes | port 2368 data packets |
| Camera `.mp4` | yes | one lens |
| IMU `.csv` | no | without it there is no velocity to compensate with, so deskew is unavailable |

Decoding a 45 s pcap takes a minute or two; progress is shown.

## What it is for

**Motion distortion.** A VLP-16 revolution takes 99.5 ms. At 19 km/h the
vehicle covers 0.53 m during one sweep, so a revolution is a shear rather
than a snapshot — points at different bearings were measured from different
positions. Projected into an image this looks like a time offset that varies
frame to frame, which is easy to mistake for a clock problem. The deskew
toggle shows the size of the effect directly, with the raw cloud left on as
a grey ghost for comparison.

The correction scales with both speed and range: a rotation of one degree
moves a point at 30 m by half a metre. So expect a large difference during
fast straight driving and during turns, and almost none at walking pace.
Measured on the 21 July clip: 0.53 m per revolution at 19 km/h, 0.06 m at
3 km/h.

**Timing.** The camera lag slider is there because the value measured for
that clip (−3.50 s) rests on a clip start time that was inferred from the
IMU and AQI logs rather than read from a manifest. Sliding it and watching
when moving objects line up is a more direct check than any correlation.

## Two things worth knowing about the data

`decode_packet` takes a `packet_time` argument. Left at its default the time
column is per-packet and spans only ~1.3 ms instead of the full 99.5 ms
revolution, which silently makes any motion compensation about 75 times too
small. This app passes it.

`filter_vel_*` in the IMU log is a **world-frame** velocity, not body-frame:
during straight driving its direction differs from `yaw_deg` by a
near-constant offset, measured at 94° on the 21 July clip. The app rotates
by `yaw + 94°` to get forward and lateral components. `filter_twist_*` is
not a usable substitute — on the same clip its magnitude was a third of the
true speed. If a different rig or session gives a different offset, POST it
to `/api/config` as `yaw_offset_deg`.

## Limits

The 3D view is the LiDAR in its own frame — it is not projected into the
image, because that needs an extrinsic. The two panes are synchronised in
time, not registered in space.

Clouds are downsampled to about 60,000 points per revolution for browser
responsiveness. The full cloud is used for the deskew computation itself.
