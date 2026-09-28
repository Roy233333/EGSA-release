# Evaluation protocol

The paper describes geometric evaluation as follows. Extract an SDF mesh using
Marching Cubes in the specified scene bounding box and sample 300,000 surface
points. Apply voxel downsampling and Statistical Outlier Removal to the sampled
reconstruction and LiDAR reference, perform coarse-to-fine point-to-plane ICP,
and retain their mutual-overlap region. Keep preprocessing, alignment and overlap
settings identical within each comparison; no smoothing is applied.

Compute unsquared Euclidean nearest-neighbour distances in both directions.
Concatenate the two distance arrays: their mean is the reported Chamfer Distance;
their median and 95th percentile are Median and P95. This pooled mean is
point-count weighted, not necessarily the equally weighted mean of the two
directional means. Convert distances to millimetres using the verified coordinate
scale. Point-to-plane distances are used for ICP, not the final table metrics.

This repository supplies the reference point clouds and a protocol summary, not
an executable geometry-evaluation workflow or per-result alignment transforms
and crop masks. Scene3 includes a full reference and a filtered main-building
reference: use the same file and spatial coverage for every compared method.
Scene4 is evaluated through RGB/normal comparisons and has no geometry GT here.

Image metrics are computed by the SDFStudio model/trainer and can be logged to
W&B or TensorBoard. W&B is the logger, not a separate metric implementation.
Specify the image indices, checkpoint and aggregation rule when reporting
results; a single-image evaluation is not a full-set average. The paper's Scene4
analysis pairs the same image at each evaluation step and averages ten paired
measurements per 50,000-iteration interval.
