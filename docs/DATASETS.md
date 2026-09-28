# Datasets and ground truth

ZIP archives are supplied separately from the training code in
[Releases](https://github.com/Roy233333/EGSA-release/releases). Extract them
outside the repository. Access currently remains restricted; code and data
licences are separate.

## Training archives

| Archive | Views | Resolution | Contents |
|---|---:|---|---|
| Scene1.zip | 328 | 384 × 384 | RGB, camera metadata, depth/normal priors, optional view pairs |
| Scene1-HR.zip | 328 | 1280 × 720 | RGB and camera metadata |
| Scene2.zip | 380 | 384 × 384 | RGB, camera metadata, depth/normal priors, NeRFBK attribution |
| Scene3.zip | 157 | 384 × 384 | RGB, camera metadata, depth/normal priors |
| Scene4.zip | 514 | 1920 × 1080 | Green-suppressed RGB, camera metadata and preprocessing notes |

Every extracted scene contains `meta_data.json`. Scene1/2/3 have RGB and prior
files in the scene directory; Scene1-HR/Scene4 store RGB under `images/`.
No training logs or checkpoints are included.

- Scene1-HR is the high-resolution runtime input for the same physical scene.
  Its camera metadata/frame correspondence is not validated as a resolution-only
  transformation of Scene1; do not assume changing image dimensions reproduces it.
- Scene1-HR and Scene4 contain no monocular priors. Disable prior loading.
- Keep optional view-pair loading disabled: Scene1 pair indexing is not validated;
  Scene2/3 specify `pairs: null`.
- Scene4 retains the experiment's green suppression, not unmodified source RGB.
- Scene2 is a processed 380-image subset, not the full 761-image source dataset.
  See [Scene2 attribution](SCENE2_ATTRIBUTION.md).

## Geometry references

| Archive | Point cloud | Points | Coverage |
|---|---|---:|---|
| Scene1-GT.zip | `reference.ply` | 68,054 | Processed concrete-block reference |
| Scene2-GT.zip | `reference.ply` | 1,116,674 | Processed NeRFBK Doss reference |
| Scene3-GT.zip | `reference.ply` | 25,657,792 | Full reference |
| Scene3-GT.zip | `evaluation_subset.ply` | 338,237 | Filtered reference retaining the main building |

Scene3's filtered version has different spatial coverage, not just lower point
density. Use the same reference and common region across compared methods.
There is no separate Scene1-HR GT package and no Scene4 geometry GT in this
distribution. Verify coordinate units and transforms before scoring; these
reference inputs are not automatically the final aligned/cropped point sets
for every result. See [evaluation](../README.md#evaluation).

## Integrity checks (optional)

`dataset_manifests/archives.sha256` contains one SHA-256 checksum per ZIP;
the other manifests list files inside each extracted scene. They contain no
experimental scores or logs. GT archives also include their own checksum list.

```powershell
Get-FileHash C:\downloads\Scene1.zip -Algorithm SHA256
python scripts/verify_dataset.py C:\datasets\Scene1 --manifest dataset_manifests/Scene1.sha256
```

Compare the ZIP hash against `archives.sha256`. For other scenes substitute
the matching directory and manifest. Checksums detect content changes or
download corruption, not camera calibration or evaluation equivalence.

## Dataset terms

Scene1, Scene3 and Scene4 are author-collected. No public dataset licence is
granted in this distribution; access is restricted and redistribution requires
the owner's permission. Scene1-HR follows Scene1's terms.

Scene2 derives from NeRFBK Doss Trento. Preserve CC BY-NC-SA 4.0 attribution,
modification notices and source terms in [SCENE2_ATTRIBUTION.md](SCENE2_ATTRIBUTION.md)
and the archive. This source licence does not establish redistribution rights
for added monocular priors; the combined archive remains access-restricted.
The code's Apache-2.0 licence does not override dataset terms.
