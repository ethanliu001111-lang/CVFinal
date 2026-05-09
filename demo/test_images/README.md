# Test Images — selection criteria

5 images covering corner cases of the 2D→3D pipeline.

| # | Filename               | Type                  | Source                          | Tests                     |
|---|------------------------|-----------------------|---------------------------------|---------------------------|
| 1 | img1_standing.jpg      | single person, frontal| Unsplash CC0                    | baseline                  |
| 2 | img2_complex_pose.jpg  | sport / yoga          | Unsplash CC0                    | pose prior                |
| 3 | img3_occluded.jpg      | half-body / occluded  | Unsplash CC0                    | in-the-wild robustness    |
| 4 | img4_multi_person.jpg  | 2-3 people interacting| Unsplash CC0                    | multi-person 3D           |
| 5 | img5_team_selfie.jpg   | self-portrait         | own / consenting roommate       | personal touch + ethics   |

**Hard requirements:**
- Unsplash photos carry the [Unsplash License](https://unsplash.com/license): free
  for commercial / non-commercial use, no permission needed; not strictly CC0
  but redistribution-safe for our academic submission.
- Resolution ≥ 800×800
- Crop all to **same aspect ratio** (1:1 or 4:3) for clean quad-plot grid
- No copyrighted images (no movie stills, no sport league screenshots, no
  classroom photos without consent)

**Selection process** (5/3 EOD task — Taijia):
1. Browse https://unsplash.com/s/photos/standing
2. Pick 4 from Unsplash + 1 self-portrait
3. Crop with ImageMagick:
   ```bash
   for f in *.jpg; do convert "$f" -gravity center -extent 1024x1024 "$f"; done
   ```
4. Commit only the 5 final images (not the source raws)
