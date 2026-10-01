# Datasets and ground truth

ZIP archives are supplied separately from the training code in
[Releases](https://github.com/Roy233333/EGSA-release/releases). Extract them
outside the repository. Code and data licences are separate; see
[dataset terms](#dataset-terms).

## Training archives

| Archive | Views | Resolution | Contents |
|---|---:|---|---|
| Scene1.zip | 328 | 384 × 384 | RGB, camera metadata, optional view pairs |
| Scene1-HR.zip | 328 | 1280 × 720 | RGB and camera metadata |
| Scene2.zip | 380 | 384 × 384 | RGB, camera metadata, NeRFBK attribution |
| Scene3.zip | 157 | 384 × 384 | RGB and camera metadata |
| Scene4.zip | 514 | 1920 × 1080 | Green-suppressed RGB, camera metadata and preprocessing notes |

Every extracted scene contains `meta_data.json`. Scene1/2/3 have RGB
files in the scene directory; Scene1-HR/Scene4 store RGB under `images/`.
No training logs or checkpoints are included.

- Scene1-HR is the high-resolution runtime input for the same physical scene.
- All distributed archives omit monocular priors. Disable prior loading unless
  you have generated and configured the required files locally.
- Keep optional view-pair loading disabled: Scene1 pair indexing is not validated;
  Scene2/3 specify `pairs: null`.
- Scene4 retains the experiment's green suppression, not unmodified source RGB.
- Scene2 is a processed 380-image subset, not the full 761-image source dataset.
  See [Scene2 attribution](SCENE2_ATTRIBUTION.md).

## Monocular priors

The depth and normal priors used for Scene1, Scene2 and Scene3 were generated
with **Omnidata v2** through SDFStudio's preprocessing workflow. These predicted
priors are not distributed in this release; they are not measured ground truth.
RGB images, frame order, camera calibration and coordinate transforms are
preserved. The distributed metadata sets `has_mono_prior` to `false` and omits
references to the excluded files.

For experiments that use these priors, obtain the models from
[Omnidata](https://github.com/EPFL-VILAB/omnidata) under their applicable terms and
generate depth and normals locally using
[SDFStudio's monocular-cue extraction script](https://github.com/autonomousvision/sdfstudio/blob/370902a10dbef08cb3fe4391bd3ed1e227b5c165/scripts/datasets/extract_monocular_cues.py).
Use the provided RGB images without repeating the crop/resize or changing camera
poses. Add each frame's `mono_depth_path` and `mono_normal_path` to
`meta_data.json`, set `has_mono_prior` to `true`, and enable prior loading only
after the files are available. Training without priors is not equivalent to the
paper's prior-augmented configurations.

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

Scene1, Scene3 and Scene4 are author-collected. No general reuse or redistribution
licence is granted for these data; contact the authors for permission.
Scene1-HR and the author-collected geometry references follow the same terms.

Scene2 derives from NeRFBK Doss Trento. Preserve CC BY-NC-SA 4.0 attribution,
modification notices and source terms in [SCENE2_ATTRIBUTION.md](SCENE2_ATTRIBUTION.md)
and the archive. The redistributed Scene2 RGB subset and reference geometry
remain subject to those source terms. Omnidata-generated priors are omitted.
The code's Apache-2.0 licence does not override dataset terms.
