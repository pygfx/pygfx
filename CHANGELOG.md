# Changelog / release notes

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Possible sections in each release:

* Added: for new features.
* Changed: for changes in existing functionality.
* Deprecated: for soon-to-be removed features.
* Removed: for now removed features.
* Fixed: for any bug fixes.
* Security: in case of vulnerabilities.


## [v0.17.0](https://github.com/pygfx/pygfx/releases/tag/v0.17.0) - 19-07-2026

### What's Changed

Fixes, nonlinear transformations, and better performance!

Fixes:

* Fix the lighting for the iso volume rendering by @almarklein in https://github.com/pygfx/pygfx/pull/1299
* Use the determinant to compute the camera sign by @gselzer in https://github.com/pygfx/pygfx/pull/1304
* View reshape by @gselzer in https://github.com/pygfx/pygfx/pull/1306
* InstancedMesh and InstancedLine now honor instance_buffer.draw_range by @almarklein in https://github.com/pygfx/pygfx/pull/1307
* Add copy=False back to reshape calls by @gselzer in https://github.com/pygfx/pygfx/pull/1308

Improvements:

* Don't use np.seterr by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1297
* Add support for nonlinear transformations via ``WorldObject.nonlinear_transform`` by @almarklein in https://github.com/pygfx/pygfx/pull/1290
* Add examples for map projections by @almarklein in https://github.com/pygfx/pygfx/pull/1291
* Better support for texture interpolation by @almarklein in https://github.com/pygfx/pygfx/pull/1300

Other:

* Add Zenodo link by @almarklein in https://github.com/pygfx/pygfx/pull/1280
* Fix discussions link in contributor guide by @almarklein in https://github.com/pygfx/pygfx/pull/1281
* Fix small typos in basics.rst by @akp2003 in https://github.com/pygfx/pygfx/pull/1293
* Update CODE_OF_CONDUCT to clarify that it applies to all repos by @almarklein in https://github.com/pygfx/pygfx/pull/1309

### New Contributors
* @akp2003 made their first contribution in https://github.com/pygfx/pygfx/pull/1293
* @gselzer made their first contribution in https://github.com/pygfx/pygfx/pull/1304

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.16.0...v0.17.0


## [v0.16.0](https://github.com/pygfx/pygfx/releases/tag/v0.16.0) - 03-03-2026

### What's Changed

Fixed:

* fix docstring formatting by @kushalkolar in https://github.com/pygfx/pygfx/pull/1234
* Fix that volume-raycast did not work with flipped camera axii by @almarklein in https://github.com/pygfx/pygfx/pull/1236
* Avoid raising an exception when upstream events have unknown keys by @almarklein in https://github.com/pygfx/pygfx/pull/1238
* Avoid snapping so hard on pixel_scale by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1240
* Fix renderer pixel scale by @almarklein in https://github.com/pygfx/pygfx/pull/1242
* No blending needed for opaque by @panxinmiao in https://github.com/pygfx/pygfx/pull/1241
* Fix TangentFrame computation by @panxinmiao in https://github.com/pygfx/pygfx/pull/1259
* Fix axis arrow colors in documentation by @butyess in https://github.com/pygfx/pygfx/pull/1262
* BF: Updated the pick location for stochastic alpha_method. by @maharshi-gor in https://github.com/pygfx/pygfx/pull/1266
* tirm indices before reshaping by @panxinmiao in https://github.com/pygfx/pygfx/pull/1271
* Fix an issue where keyframe optimization was not applied correctly. by @panxinmiao in https://github.com/pygfx/pygfx/pull/1269
* Improve SkinnedMesh bbox again by @panxinmiao in https://github.com/pygfx/pygfx/pull/1263
* Fix GPU object hash collisions after GC by @limdaepl in https://github.com/pygfx/pygfx/pull/1265
* Prevent flicker if volume slice is exactly in between two slices by @almarklein in https://github.com/pygfx/pygfx/pull/1278

Changed:

* Make `render_order` independent of scene graph and introduce "group order" by @panxinmiao in https://github.com/pygfx/pygfx/pull/1222
* Improve Bounding Box Computation for SkinnedMesh by @panxinmiao in https://github.com/pygfx/pygfx/pull/1258
* Use new wgpu default device by @almarklein in https://github.com/pygfx/pygfx/pull/1275
* Use new wgpu structs by @almarklein in https://github.com/pygfx/pygfx/pull/1199

Added:

* KHR_draco_mesh_compression: support generic attributes by @panxinmiao in https://github.com/pygfx/pygfx/pull/1224
* glTF: support accessor sparse by @panxinmiao in https://github.com/pygfx/pygfx/pull/1223
* Add  `environment`   Property to `Scene` by @panxinmiao in https://github.com/pygfx/pygfx/pull/1156
* Support KHR_materials_pbrSpecularGlossiness by @panxinmiao in https://github.com/pygfx/pygfx/pull/1230
* Physically Based Bloom by @panxinmiao in https://github.com/pygfx/pygfx/pull/1227
* Add uniform.maprange to (almost) all objects that have .map by @almarklein in https://github.com/pygfx/pygfx/pull/1246
* Fix picking index for volume by @almarklein in https://github.com/pygfx/pygfx/pull/1253
* Post-processing: normal reconstruction by @Korijn in https://github.com/pygfx/pygfx/pull/1254
* Add UV transform support for the sheen map by @panxinmiao in https://github.com/pygfx/pygfx/pull/1268

Other:

* Allow newer wgpu-versions by @almarklein in https://github.com/pygfx/pygfx/pull/1229
* New lower bounds for rendercanvas and wgpu by @almarklein in https://github.com/pygfx/pygfx/pull/1235
* Bump dependencies by @almarklein in https://github.com/pygfx/pygfx/pull/1276
* Update contributor guide and add AI policy by @almarklein in https://github.com/pygfx/pygfx/pull/1277

### New Contributors
* @butyess made their first contribution in https://github.com/pygfx/pygfx/pull/1262
* @maharshi-gor made their first contribution in https://github.com/pygfx/pygfx/pull/1266

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.15.0...v0.16.0


## [v0.15.3](https://github.com/pygfx/pygfx/releases/tag/v0.15.3) - 15-12-2025

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.15.2...v0.15.3


## [v0.15.2](https://github.com/pygfx/pygfx/releases/tag/v0.15.2) - 11-11-2025

A hotfix that updates the upper bounds for dependencies `rendercanvas`  and `wgpu-py`.

This is not a reflection of `main` at the time of release, but basically `v0.15.0` with updated dependency bounds.

**Full Changelog compared to 0.15.0**: https://github.com/pygfx/pygfx/compare/v0.15.0...v0.15.2


## [v0.15.0](https://github.com/pygfx/pygfx/releases/tag/v0.15.0) - 30-10-2025

### What's Changed
* fix typo in marker exception message by @kushalkolar in https://github.com/pygfx/pygfx/pull/1214
* remove assert in sprite, some cleanup by @kushalkolar in https://github.com/pygfx/pygfx/pull/1216
* Add six-legged asterisk marker by @almarklein in https://github.com/pygfx/pygfx/pull/1219
* account for transform of instanced objects for calculating bbox by @kushalkolar in https://github.com/pygfx/pygfx/pull/1220
* use `pylinalg>=0.6.8,<7.0` by @kushalkolar in https://github.com/pygfx/pygfx/pull/1225


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.14.0...v0.15.0


## [v0.14.0](https://github.com/pygfx/pygfx/releases/tag/v0.14.0) - 09-10-2025

### What's Changed

Added:

* Per-vertex markers by @almarklein in https://github.com/pygfx/pygfx/pull/1208 and https://github.com/pygfx/pygfx/pull/1211
* Support for proper tick marks by adding a 'tick' marker and rotation_mode 'curve' by @almarklein in https://github.com/pygfx/pygfx/pull/1209
* Add Eye Dome Lighting (EDL) EffectPass by @mortacious in https://github.com/pygfx/pygfx/pull/1200

Changed:

* Remove last use of wgpu.gui by @almarklein in https://github.com/pygfx/pygfx/pull/1198
* Update for new imgui by @almarklein in https://github.com/pygfx/pygfx/pull/1196
* Use font-awesome again in 2 examples by @almarklein in https://github.com/pygfx/pygfx/pull/1201
* Encode face coords as 2x9 bits instead of 3x6 bits by @limdaepl in https://github.com/pygfx/pygfx/pull/1206
* Update ssaa.wgsl and ddaa.wgsl by @almarklein in https://github.com/pygfx/pygfx/pull/1207

Other:

* Adjust to new Ruff by @almarklein in https://github.com/pygfx/pygfx/pull/1195
* Allow up to wgpu 0.25 by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1205
* No upper bounds on versions of dependencies by @almarklein in https://github.com/pygfx/pygfx/pull/1212

### New Contributors
* @mortacious made their first contribution in https://github.com/pygfx/pygfx/pull/1200

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.13.0...v0.14.0


## [v0.13.0](https://github.com/pygfx/pygfx/releases/tag/v0.13.0) - 05-09-2025

### Introduction

This release represents a lot of work on refactoring how PyGfx deals with transparency. We recommend reading https://docs.pygfx.org/latest/transparency.html to learn how PyGfx works now.

The most important changes related to transparency:

The `renderer.blend_mode` is no more. Some modes like the 'ordered2' and 'weighted_plus' no longer exist. Instead, you can now set `material.alpha_mode` on each object separately. Yes, you can mix different modes in the same scene.

It does mean that as a user you'll have to be a bit more aware of whether your objects are transparent, and how to approach that. In short, the default `material.alpha_mode='auto'` works fine for opaque and most semi-transparent objects, as well as for objects that have mixed opaque/transparent fragments. But if you know your object is opaque, use `alpha_mode='solid'`, and if it's transparent, use `alpha_mode='blend'`. For more context check https://docs.pygfx.org/latest/transparency.html#a-quick-guide-to-select-alpha-mode

We also changed some things on how we approach anti-aliasing:

* The SSAA (super-sample anti-aliasing) that we already applied now uses an improved filter (the Mitchell filter). The same improved filter is also used for upsampling. Also see the new ``renderer.pixel_scale`` and ``renderer.pixel_filter``.
* We added support for PPAA (post-processing anti-aliasing) to improve the smoothness of the rendered result. It's enabled by default. See the new `renderer.ppaa`.
* The `material.aa` for text lines and points is now off by default, to prevent blending artifacts. Though we recommend turning it on for text for higher quality glyphs.
* See https://docs.pygfx.org/latest/aa.html for details.

### API changes

Objects and materials:

- Removed `WorldObject.render_mask`.
- Added `Material.alpha_mode`, default is 'auto'. 
- Adeded `.alpha_config`, `.alpha_method` (setting `alpha_mode` also sets these).
- Added `Material.render_queue`, default is None (auto). 
- Added `Material.depth_write`, default is None (auto).  
- Added `Material.depth_compare, default is "<".
- Added `Material.alpha_test`.
- Added `Material.alpha_compare`, default is "<".
- Removed `material.color_is_transparent` for `Line`, `Mesh`, `Points`, `Text`. Use `material.color.a < 1` instead.
- Removed `material.edge_color_is_transparent` for `Points`. Use `material.edge_color.a < 1` instead.
- Removed `material.outline_color_is_transparent` for `Text`. Use `material.outline_color.a < 1` instead.

Renderer:
- Removed `Renderer.blend_mode`.
- The `renderer.sort_objects` is True by default.
- The renderer's sorting logic is completely overhauled.
- Added `renderer.effect_passes`.
- Added `renderer.ppaa`.
- Added `renderer.pixel_scale`.
- The `renderer.pixel_filter` is now a string instead of an int.
- Added `renderer.clear()` to clear render targets.

### Commits

### Changes related to transparency and anti-aliasing

* Simplify blending by @almarklein in https://github.com/pygfx/pygfx/pull/1002
* Add support for post-processing effects by @almarklein in https://github.com/pygfx/pygfx/pull/1114
* Improve the interpolation filter in flush by @almarklein in https://github.com/pygfx/pygfx/pull/1121
* Fix that snapshot may not include the last effect by @almarklein in https://github.com/pygfx/pygfx/pull/1126
* Syncing ssaa.wgsl by @almarklein in https://github.com/pygfx/pygfx/pull/1131
* Improve dithered alpha_mode by @almarklein in https://github.com/pygfx/pygfx/pull/1139
* Add support for FXAA and DDAA by @almarklein in https://github.com/pygfx/pygfx/pull/1117
* remove unused variable in image_stitching example by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1143
* Add docs for blending by @almarklein in https://github.com/pygfx/pygfx/pull/1124
* Implement material.alpha_mode to replace .blending and .transparent by @almarklein in https://github.com/pygfx/pygfx/pull/1144
* Allow calling renderer.clear(depth=True, color=True) by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1170
* Rename  alpha-method 'composite' to 'blended' by @almarklein in https://github.com/pygfx/pygfx/pull/1172
* Add renderer.pixel_scale by @almarklein in https://github.com/pygfx/pygfx/pull/1174
* Turn off material.aa by default by @almarklein in https://github.com/pygfx/pygfx/pull/1171
* Enable weighted blending between calls to render by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1148
* Add docs on aa by @almarklein in https://github.com/pygfx/pygfx/pull/1173
* Make background use alpha_mode and render_queue correctly by @almarklein in https://github.com/pygfx/pygfx/pull/1180
* Add material.alpha_method (readonly) by @almarklein in https://github.com/pygfx/pygfx/pull/1184
* Refactor logic related to 'auto' mode by @almarklein in https://github.com/pygfx/pygfx/pull/1186
* Tweak docs by @almarklein in https://github.com/pygfx/pygfx/pull/1188
* Add arg to Ruler for alpha_mode of child objects by @almarklein in https://github.com/pygfx/pygfx/pull/1189
* Remove that material.opacity affects alpha logic by @almarklein in https://github.com/pygfx/pygfx/pull/1190
* Small tweak to docs of render_queue by @almarklein in https://github.com/pygfx/pygfx/pull/1194

### Changes to the engine

* Add support for custom orbit center in OrbitController by @panxinmiao in https://github.com/pygfx/pygfx/pull/1115
* Cloth Simulation with Compute Shaders by @panxinmiao in https://github.com/pygfx/pygfx/pull/1116
* Fix that the ruler did not take world transform into account by @almarklein in https://github.com/pygfx/pygfx/pull/1130
* update roll by @sawbz in https://github.com/pygfx/pygfx/pull/1134
* Fix pan in orbit controller by @almarklein in https://github.com/pygfx/pygfx/pull/1135
* fix GLTFMaterialsSheenExtension by @panxinmiao in https://github.com/pygfx/pygfx/pull/1150
* Fixed some edge cases in the Interpolant by @panxinmiao in https://github.com/pygfx/pygfx/pull/1154
* Implement GpuCache with LRU strategy (split from #974) by @panxinmiao in https://github.com/pygfx/pygfx/pull/1142
* Add `EXT_texture_webp` support by @panxinmiao in https://github.com/pygfx/pygfx/pull/1155
* Support `KHR_draco_mesh_compression` by @panxinmiao in https://github.com/pygfx/pygfx/pull/1159
* Fix keyframe optimization by @panxinmiao in https://github.com/pygfx/pygfx/pull/1166
* Fix docstring of get_bounding_box by @almarklein in https://github.com/pygfx/pygfx/pull/1178
* Make sure the pick face coords add up to one by @almarklein in https://github.com/pygfx/pygfx/pull/1179
* Support GLTF alphaMode by @almarklein in https://github.com/pygfx/pygfx/pull/1177
* Dereference world-object in case its a weakproxy by @almarklein in https://github.com/pygfx/pygfx/pull/1183
* Can use wgpu 0.23 by @almarklein in https://github.com/pygfx/pygfx/pull/1193

### Other changes

* Use rendercanvas in all our examples by @almarklein in https://github.com/pygfx/pygfx/pull/1127
* Minor tweaks to avoid warnings in CI by @almarklein in https://github.com/pygfx/pygfx/pull/1129
* Fix vertical text because uharfbuzz 0.51 broke something by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1151
* Update uharfbuzz constraint for future proofing by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1152
* Adjust how version is reported when one checkouts a tag by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1153
* Update validation example screenshot by @almarklein in https://github.com/pygfx/pygfx/pull/1161
* updated "expected 1.0" to 1 year from today by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1160
* Add dedicated email address to coc by @almarklein in https://github.com/pygfx/pygfx/pull/1181
* Update _version.py by @almarklein in https://github.com/pygfx/pygfx/pull/1185
* Add slotscheck to CI by @almarklein in https://github.com/pygfx/pygfx/pull/1187
* Switch to trusted publishing for PyPI publish in CI by @EpicWink in https://github.com/pygfx/pygfx/pull/1192

### New Contributors

* @sawbz made their first contribution in https://github.com/pygfx/pygfx/pull/1134
* @EpicWink made their first contribution in https://github.com/pygfx/pygfx/pull/1192

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.12.0...v0.13.0


## [v0.12.0](https://github.com/pygfx/pygfx/releases/tag/v0.12.0) - 12-06-2025

### What's Changed
* Add option to ScreenCoordsCamera to invert y by @almarklein in https://github.com/pygfx/pygfx/pull/1092
* Ray tracing example by @panxinmiao in https://github.com/pygfx/pygfx/pull/1068
* Support texture transform by @panxinmiao in https://github.com/pygfx/pygfx/pull/999


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.11.0...v0.12.0


## [v0.11.0](https://github.com/pygfx/pygfx/releases/tag/v0.11.0) - 04-06-2025

### What's Changed

### Some more improvements related to camera and depth

* Camera with fov==0 does not move backwards when zooming out by @almarklein in https://github.com/pygfx/pygfx/pull/1090
* Set ruler's z-position closer to origin to avoid ruler disappearing when camera.depth is small by @almarklein in https://github.com/pygfx/pygfx/pull/1101

### User-facing changes

* Drop Python 3.9 because its already broken, and bump CIs by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1104
* Pass through worldobject parameters from Text Object by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1102
* Promote the use of VolumeMipMaterial instead of the VolumeRayMaterial base class by @almarklein in https://github.com/pygfx/pygfx/pull/1107

### Internal (and low-level) changes

* fix pre-computed tangents by @panxinmiao in https://github.com/pygfx/pygfx/pull/1088
* Uniform buffers can now handle vec3 and matnx3 by @almarklein in https://github.com/pygfx/pygfx/pull/1091
* Fix matnx3 fields in uniforms by @almarklein in https://github.com/pygfx/pygfx/pull/1094
* Migrate to `logger.warning` by @emmanuel-ferdman in https://github.com/pygfx/pygfx/pull/1098

### New Contributors

* @emmanuel-ferdman made their first contribution in https://github.com/pygfx/pygfx/pull/1098

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.10.0...v0.11.0


## [v0.10.0](https://github.com/pygfx/pygfx/releases/tag/v0.10.0) - 19-05-2025

### What's Changed

### Camera's and controllers

* Add `match_aspect` arg to camera.show_object() by @almarklein in https://github.com/pygfx/pygfx/pull/1051
* Change controller.add_camera so it does not change camera order by @almarklein in https://github.com/pygfx/pygfx/pull/1086
* add `Controller.pause` context manager by @kushalkolar in https://github.com/pygfx/pygfx/pull/1080
* `Camera.get_state()` returns new copy of the array instead of a view by @kushalkolar in https://github.com/pygfx/pygfx/pull/1076
* Improvements to cameras by @almarklein in https://github.com/pygfx/pygfx/pull/1085
  * Added `camera.depth` that gets automaticallet by `camera.show_()` methods. Can be manually set in cases where the depth is very different from width and height. 
  * Avoid depth clip issues when zooming in by calculating depth range using `depth`..
  * The `Ruler` no longer reports a bounding box, so that axii in the scene no longer break `camera.show_object()`.
  * The OrthographicCamera no longer move in the view direction (typically z-axis) as it zooms. Note that a PerspectiveCamera with `fov==0` still does.

### Rendering

* Add support for loops to line object by @almarklein in https://github.com/pygfx/pygfx/pull/1056
* Add LineInfSegmentMaterial for infinite lines by @almarklein in https://github.com/pygfx/pygfx/pull/1072
* Refactor engine's resolving of fragment shader output  by @almarklein in https://github.com/pygfx/pygfx/pull/1053
* Fix pick info inheritance by @limdaepl in https://github.com/pygfx/pygfx/pull/1057
* Instanced lines by @limdaepl in https://github.com/pygfx/pygfx/pull/1058

### Examples

* Create background subtraction example by @kingbedjed in https://github.com/pygfx/pygfx/pull/1067
* Example to compute histogram using a compute shader by @almarklein in https://github.com/pygfx/pygfx/pull/1063

### Other

* Small cleanups for the CI pipeline by @berendkleinhaneveld in https://github.com/pygfx/pygfx/pull/1061
* Don't throw AttributeError when calling dict attributes by @berendkleinhaneveld in https://github.com/pygfx/pygfx/pull/1062
* Make the Store track hashable objects by value by @almarklein in https://github.com/pygfx/pygfx/pull/1066
* Properly implement `register_wgsl_loader` by @almarklein in https://github.com/pygfx/pygfx/pull/1083
* Increase upper bound to wgpu version by @almarklein in https://github.com/pygfx/pygfx/pull/1074


### New Contributors
* @limdaepl made their first contribution in https://github.com/pygfx/pygfx/pull/1057
* @kingbedjed made their first contribution in https://github.com/pygfx/pygfx/pull/1067

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.9.0...v0.10.0


## [v0.9.0](https://github.com/pygfx/pygfx/releases/tag/v0.9.0) - 19-03-2025

### API changes

The biggest change in the release is that we made the `geometry` a simple object to contain buffers and textures. It does not have any methods, and you should probably not subclass it.

* Move Text API and implementation to Text world object by @almarklein in https://github.com/pygfx/pygfx/pull/1034
  * `TextGeometry` and `MultiTextGeometry` are gone. Use `Text` and `MultiText` instead. 
  * See https://github.com/pygfx/pygfx/pull/1034 for a small migration guide.
* Remove `get_bounding_box()` from geometry by @almarklein in https://github.com/pygfx/pygfx/pull/1042
* Make geometry a simple Store by @almarklein in https://github.com/pygfx/pygfx/pull/1044

### Behavioral changes
* Change colormap interpolation to be consistent by @almarklein in https://github.com/pygfx/pygfx/pull/1025

### New features
* Support marker rotation by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1027
* Support for custom colormaps, and lazy loading by @almarklein in https://github.com/pygfx/pygfx/pull/1031
* Support “KHR_lights_punctual” by @panxinmiao in https://github.com/pygfx/pygfx/pull/1030
* Add Anisotropy support by @panxinmiao in https://github.com/pygfx/pygfx/pull/1037
* Add Sheen support by @panxinmiao in https://github.com/pygfx/pygfx/pull/1039

### Other changes
* Allow wgpu 0.19 - 0.21 by @almarklein in https://github.com/pygfx/pygfx/pull/1043
* Improve tracking of bindings by @almarklein in https://github.com/pygfx/pygfx/pull/1026
* Unified mesh texture map logic and special treatment for colormaps by @panxinmiao in https://github.com/pygfx/pygfx/pull/1008
* Rename internal define_colormap() func for clarity by @almarklein in https://github.com/pygfx/pygfx/pull/1032
* Early exit in buffer/texture update_xx methods by @almarklein in https://github.com/pygfx/pygfx/pull/1033
* Refactor IBL related code by @panxinmiao in https://github.com/pygfx/pygfx/pull/1035
* Fix and enhance normal map and tangents by @panxinmiao in https://github.com/pygfx/pygfx/pull/1036
* Add back the text "dither" in the validation example by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1040
* Fix test on windows by @almarklein in https://github.com/pygfx/pygfx/pull/1038
* No longer set the Bone object as invisible by default by @panxinmiao in https://github.com/pygfx/pygfx/pull/1041
* Fix morph on MacOS by @almarklein in https://github.com/pygfx/pygfx/pull/1046
* Small typo fix by @kevinyamauchi in https://github.com/pygfx/pygfx/pull/1050

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.8.0...v0.9.0


## [v0.8.0](https://github.com/pygfx/pygfx/releases/tag/v0.8.0) - 06-03-2025

### Summary

It's been a while, so quite a lot of changes! The gltf support has been improved a lot, the transform system has been optimized for performance, `gamma` prop for image and volume materials, improved text rendering (inc. multi-text), faster clipping planes, and much more!

### API changes

* Added `ImageBasicMaterial.gamma` and `VolumeBasicMaterial.gamma`.
* Added `PointsMarkerMaterial.edge_color_mode`.
* Added `MeshPhysicalMaterial.iridescence`, `.iridescence_ior`, and `.iridescence_thickness_range`.
* Added `MultiTextGeometry` and `TextBlock` for efficiently handling multiple texts (e.g. labels).
* Added `TextGeometry.max_width`, `.paragraph_spacing`, added Markdown support for headers and bullets, and support for vertical text.

### PRs merged
* Tweak caching of shadow pipeline by @almarklein in https://github.com/pygfx/pygfx/pull/935
* Implement GLTFCubicSplineInterpolant by @panxinmiao in https://github.com/pygfx/pygfx/pull/936
* Refactor environment and pipeline by @almarklein in https://github.com/pygfx/pygfx/pull/912
* Add MeshPhysicalMaterial with Clearcoat, Specular, and IOR Support by @panxinmiao in https://github.com/pygfx/pygfx/pull/908
* Associate pipeline-container also with Material by @almarklein in https://github.com/pygfx/pygfx/pull/939
* Add `package_name` as an argument for `load_wgsl` by @m-agour in https://github.com/pygfx/pygfx/pull/943
* Fix ior & specular not work correctly by @panxinmiao in https://github.com/pygfx/pygfx/pull/942
* Refactor the plugin extension mechanism of the glTF loader by @panxinmiao in https://github.com/pygfx/pygfx/pull/940
* Cache camera matrices by @Korijn in https://github.com/pygfx/pygfx/pull/945
* Fix CI by pinning trimesh by @almarklein in https://github.com/pygfx/pygfx/pull/952
* Tweak some docstrings by @almarklein in https://github.com/pygfx/pygfx/pull/949
* Optimize last_modified tracking by @Korijn in https://github.com/pygfx/pygfx/pull/950
* Refactor to have a single scene traverse in the renderer by @almarklein in https://github.com/pygfx/pygfx/pull/953
* Tweak the method names that update world objects by @almarklein in https://github.com/pygfx/pygfx/pull/955
* Make `bind_matrix` a read-only property, and clean up `np.linalg.inv`. by @panxinmiao in https://github.com/pygfx/pygfx/pull/948
* Auto update skeleton and skeleton_helper by @panxinmiao in https://github.com/pygfx/pygfx/pull/894
* decrease minimum version of imgui-bundle by @hmaarrfk in https://github.com/pygfx/pygfx/pull/958
* Add  more details on drivers in install instructions by @almarklein in https://github.com/pygfx/pygfx/pull/959
* add gamma to image and volume by @tlambert03 in https://github.com/pygfx/pygfx/pull/957
* Add typing to `objects._base` by @tlambert03 in https://github.com/pygfx/pygfx/pull/960
* 🚀 RecursiveTransform optimization by @Korijn in https://github.com/pygfx/pygfx/pull/963
* Add typing to Geometry and Material by @tlambert03 in https://github.com/pygfx/pygfx/pull/961
* Update CI badge by @Korijn in https://github.com/pygfx/pygfx/pull/964
* ⚡ Micro optimization by @Korijn in https://github.com/pygfx/pygfx/pull/965
* ⚡⚡ Micro-optimized flag_update again by @Korijn in https://github.com/pygfx/pygfx/pull/969
* Improve z fighting control for text by @hmaarrfk in https://github.com/pygfx/pygfx/pull/968
* Add support for edge_colors on a per vertex for Points by @hmaarrfk in https://github.com/pygfx/pygfx/pull/970
* Provide a path for end users to register their own blenders by @hmaarrfk in https://github.com/pygfx/pygfx/pull/966
* Add __slots__ to transform objects by @Korijn in https://github.com/pygfx/pygfx/pull/971
* Reduce intermediate array usage in Transform by @Korijn in https://github.com/pygfx/pygfx/pull/973
* Set writeable false in appropriate places, clean up a few computations by @Korijn in https://github.com/pygfx/pygfx/pull/977
* Fix typo in guide.rst by @boisgera in https://github.com/pygfx/pygfx/pull/983
* No need to divide by the length of a unit vector in grid by @hmaarrfk in https://github.com/pygfx/pygfx/pull/984
* Add Iridescence support to MeshPhysicalMaterial by @panxinmiao in https://github.com/pygfx/pygfx/pull/980
* Components setter & animation optimization by @Korijn in https://github.com/pygfx/pygfx/pull/978
* Stabilize ruler step calculation to avoid grid flicker by @almarklein in https://github.com/pygfx/pygfx/pull/986
* Colormap clamp by @almarklein in https://github.com/pygfx/pygfx/pull/987
* Improve performance of wobject.world.position by @Korijn in https://github.com/pygfx/pygfx/pull/988
* Add glTF Extension: "KHR_materials_emissive_strength" by @panxinmiao in https://github.com/pygfx/pygfx/pull/994
* Use correct diffuse color by @panxinmiao in https://github.com/pygfx/pygfx/pull/991
* Fix clearcoat_roughness_map channel by @panxinmiao in https://github.com/pygfx/pygfx/pull/997
* Micro optimize video decoding by @hmaarrfk in https://github.com/pygfx/pygfx/pull/993
* Generate mipmaps using RenderPipline and linear interpolation downsampling by @panxinmiao in https://github.com/pygfx/pygfx/pull/995
* Add "KHR_materials_unlit" support by @panxinmiao in https://github.com/pygfx/pygfx/pull/1004
* Multitext and text refactoring by @almarklein in https://github.com/pygfx/pygfx/pull/834
* Update gitignore by @panxinmiao in https://github.com/pygfx/pygfx/pull/1009
* Text performance tweaks by @almarklein in https://github.com/pygfx/pygfx/pull/1006
* Use single buffer to store per-glyph data by @almarklein in https://github.com/pygfx/pygfx/pull/1010
* Fix waterfall example by @almarklein in https://github.com/pygfx/pygfx/pull/1012
* Refactor punctual light shading code by @panxinmiao in https://github.com/pygfx/pygfx/pull/1016
* Prevent glyphs from bleeding into neighbours in atlas by @almarklein in https://github.com/pygfx/pygfx/pull/1014
* Fix for latest trimesh by @almarklein in https://github.com/pygfx/pygfx/pull/1017
* Fix TransformGizmo.set_object when object is `None` by @schlegelp in https://github.com/pygfx/pygfx/pull/1020
* Add image comparator example by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1019
* Improve clipping planes by @panxinmiao in https://github.com/pygfx/pygfx/pull/1022
* fix two bugs in atlas allocator by @hmaarrfk in https://github.com/pygfx/pygfx/pull/1021
* Ruler use multitext by @almarklein in https://github.com/pygfx/pygfx/pull/1018
* Bump version by @almarklein in https://github.com/pygfx/pygfx/pull/1023

### New Contributors
* @m-agour made their first contribution in https://github.com/pygfx/pygfx/pull/943
* @boisgera made their first contribution in https://github.com/pygfx/pygfx/pull/983

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.7.0...v0.8.0


## [v0.7.0](https://github.com/pygfx/pygfx/releases/tag/v0.7.0) - 14-01-2025

### What's Changed
* Support for loading Open3D triangle meshes, materials and pointclouds into pygfx by @cansik in https://github.com/pygfx/pygfx/pull/885
* Lint Ruff 0.8.0 and bump imgui-bundle version to match wgpu-py by @claydugo in https://github.com/pygfx/pygfx/pull/890
* Add SAM2 point segmentation example by @claydugo in https://github.com/pygfx/pygfx/pull/876
* implement suggested improvements listed in issue343 by @zetyquickly in https://github.com/pygfx/pygfx/pull/887
* an example of how YUV colorspace can be implemented by @hmaarrfk in https://github.com/pygfx/pygfx/pull/873
* Add internal util to easily check arg types by @almarklein in https://github.com/pygfx/pygfx/pull/897
* Fix that send_data could not set  multiple layers at once by @almarklein in https://github.com/pygfx/pygfx/pull/898
* Add an example of a tiny image -- 1x1 pixel by @hmaarrfk in https://github.com/pygfx/pygfx/pull/900
* Enable using rendercanvas by @almarklein in https://github.com/pygfx/pygfx/pull/899
* Fix roughness by @panxinmiao in https://github.com/pygfx/pygfx/pull/903
* Implement a dither blend mode by @almarklein in https://github.com/pygfx/pygfx/pull/902
* Animation mixer by @panxinmiao in https://github.com/pygfx/pygfx/pull/892
* Implement colorspace tex-srgb by @almarklein in https://github.com/pygfx/pygfx/pull/906
* Introduce TextureMap by @panxinmiao in https://github.com/pygfx/pygfx/pull/889
* Bump pylinalg by @Korijn in https://github.com/pygfx/pygfx/pull/915
* Ensure Transforms are evaluated lazily by @Korijn in https://github.com/pygfx/pygfx/pull/911
* Default TextureMap settting by @panxinmiao in https://github.com/pygfx/pygfx/pull/918
* Base AffineTransform state on position/scale/rotation instead of matrix by @Korijn in https://github.com/pygfx/pygfx/pull/916
* Add state_basis option to AffineTransform to toggle between position/scale/rotation or matrix as basis of state by @Korijn in https://github.com/pygfx/pygfx/pull/920
* Improve TextureMap docs by @almarklein in https://github.com/pygfx/pygfx/pull/924
* Bump pylinalg and use mat_has_shear by @Korijn in https://github.com/pygfx/pygfx/pull/927
* Print out the jijna'ed WGSL code to help debug by @hmaarrfk in https://github.com/pygfx/pygfx/pull/917
* Matrix inverse by @panxinmiao in https://github.com/pygfx/pygfx/pull/919
* Support taking hirez screenshots by @almarklein in https://github.com/pygfx/pygfx/pull/909
* Bump pylinalg & ruff format by @Korijn in https://github.com/pygfx/pygfx/pull/931
* Add Remote URL Support to glTF loader by @panxinmiao in https://github.com/pygfx/pygfx/pull/895
* Fix the issue where stopping an AnimationAction did not restore the target to its initial state. by @panxinmiao in https://github.com/pygfx/pygfx/pull/933
* Bump version by @almarklein in https://github.com/pygfx/pygfx/pull/934

### New Contributors
* @zetyquickly made their first contribution in https://github.com/pygfx/pygfx/pull/887

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.6.0...v0.7.0


## [v0.6.0](https://github.com/pygfx/pygfx/releases/tag/v0.6.0) - 15-11-2024

### What's Changed
* Implement set_wgpu_limits by @almarklein in https://github.com/pygfx/pygfx/pull/860
* Use pyproject.toml and ruff by @almarklein in https://github.com/pygfx/pygfx/pull/859
* Get a longer version string on dev installs by @almarklein in https://github.com/pygfx/pygfx/pull/861
* Do not mix stdout and stderr in version detection by @hmaarrfk in https://github.com/pygfx/pygfx/pull/862
* KeyframeTrack and Interpolants by @panxinmiao in https://github.com/pygfx/pygfx/pull/756
* Fix install of mesa drivers by @almarklein in https://github.com/pygfx/pygfx/pull/869
* Tweak _version.py a little by @almarklein in https://github.com/pygfx/pygfx/pull/870
* Allow turning features off by @almarklein in https://github.com/pygfx/pygfx/pull/868
* Add labelled image grid example by @claydugo in https://github.com/pygfx/pygfx/pull/867
* add intersphinx to docs by @kushalkolar in https://github.com/pygfx/pygfx/pull/871
* Implement texture/buffer send_data() by @almarklein in https://github.com/pygfx/pygfx/pull/879
* Tweaks to readme and docs by @almarklein in https://github.com/pygfx/pygfx/pull/880
* Fix that image shader was not invalided when map is set to None by @almarklein in https://github.com/pygfx/pygfx/pull/884
* Adjust to new wgpu-py by @almarklein in https://github.com/pygfx/pygfx/pull/878
* Bump version by @almarklein in https://github.com/pygfx/pygfx/pull/886

### New Contributors
* @claydugo made their first contribution in https://github.com/pygfx/pygfx/pull/867

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.5.0...v0.6.0


## [v0.5.0](https://github.com/pygfx/pygfx/releases/tag/v0.5.0) - 25-09-2024

### What's Changed
* Use default min_binding_size by @almarklein in https://github.com/pygfx/pygfx/pull/856
* Allow users to customize which edge of the SDF is painted by @hmaarrfk in https://github.com/pygfx/pygfx/pull/824
* Use vstack and not row_stack by @hmaarrfk in https://github.com/pygfx/pygfx/pull/857
* Add morph targets support. by @panxinmiao in https://github.com/pygfx/pygfx/pull/751
* A more perfect SDF for the pin by @hmaarrfk in https://github.com/pygfx/pygfx/pull/837
* Provide tools to debug SDFs by @hmaarrfk in https://github.com/pygfx/pygfx/pull/823
* Improve Skeleton's transform performance by @panxinmiao in https://github.com/pygfx/pygfx/pull/746


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.4.1...v0.5.0


## [v0.4.1](https://github.com/pygfx/pygfx/releases/tag/v0.4.1) - 25-09-2024

### What's Changed
* fix typo in test suite comment by @hmaarrfk in https://github.com/pygfx/pygfx/pull/849
* Enhance and fix materials by @panxinmiao in https://github.com/pygfx/pygfx/pull/850
* Fix binding layout by @panxinmiao in https://github.com/pygfx/pygfx/pull/852
* Use new repo of gltf assets by @almarklein in https://github.com/pygfx/pygfx/pull/854


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.4.0...v0.4.1


## [v0.4.0](https://github.com/pygfx/pygfx/releases/tag/v0.4.0) - 11-09-2024

### What's Changed
* Improve spelling for render_mask in blender.py by @hmaarrfk in https://github.com/pygfx/pygfx/pull/801
* Remove unreachable return value by @hmaarrfk in https://github.com/pygfx/pygfx/pull/802
* Refactor mesh shading code by @panxinmiao in https://github.com/pygfx/pygfx/pull/800
* nit: ensure that pos_offset_x is always defined even if mistake by dev by @hmaarrfk in https://github.com/pygfx/pygfx/pull/805
* Sdf clipping planes by @hmaarrfk in https://github.com/pygfx/pygfx/pull/804
* Add the name property to more objects by @hmaarrfk in https://github.com/pygfx/pygfx/pull/807
* Add Ruler object by @almarklein in https://github.com/pygfx/pygfx/pull/787
* Fix that wgpu-py requires a blend-dict now by @almarklein in https://github.com/pygfx/pygfx/pull/816
* Fixed that dashing was not applied to thin lines by @almarklein in https://github.com/pygfx/pygfx/pull/811
* Cleanup the point shader to enable people to better customize it by @hmaarrfk in https://github.com/pygfx/pygfx/pull/822
* Example to demo differences between the image shader and mesh by @hmaarrfk in https://github.com/pygfx/pygfx/pull/820
* Allow a minimum number of fonts to always be cache by @hmaarrfk in https://github.com/pygfx/pygfx/pull/817
* Custom point marker SDF enabling logo insertion by @hmaarrfk in https://github.com/pygfx/pygfx/pull/789
* unproject to get range of screen space to set rulers by @kushalkolar in https://github.com/pygfx/pygfx/pull/814
* Buffer chunking and more by @almarklein in https://github.com/pygfx/pygfx/pull/795
* Fix pointer down interaction in stent demo volume_render1 by @hmaarrfk in https://github.com/pygfx/pygfx/pull/826
* Fix infgrid artifacts for ortho camera by @almarklein in https://github.com/pygfx/pygfx/pull/829
* Update internal usage of buffers and textures by @almarklein in https://github.com/pygfx/pygfx/pull/828
* Improve support for custom tick formatting by @almarklein in https://github.com/pygfx/pygfx/pull/831
* Small change in spelling by @almarklein in https://github.com/pygfx/pygfx/pull/833
* Improve local testing by @hmaarrfk in https://github.com/pygfx/pygfx/pull/827
* Fix pick_write not really being dynamic by @almarklein in https://github.com/pygfx/pygfx/pull/842
* Update to latest and upcoming wgpu-py by @almarklein in https://github.com/pygfx/pygfx/pull/848
* Add MeshToonMaterial by @panxinmiao in https://github.com/pygfx/pygfx/pull/803


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.3.0...v0.4.0


## [v0.3.0](https://github.com/pygfx/pygfx/releases/tag/v0.3.0) - 20-06-2024



### Changes
* Change event time_stamp unit to seconds by @panxinmiao in https://github.com/pygfx/pygfx/pull/742
  * Check your code for usage of `event["time_stamp"]`. 
* Move text slightly in front of the outline. by @hmaarrfk in https://github.com/pygfx/pygfx/pull/776
  * Text can now be placed *on* a surface.
  * The text outline no longer bleeds over the face of neighboring characters. 
* More flexible linking controllers to cameras by @almarklein in https://github.com/pygfx/pygfx/pull/778
  * `PerspectiveCamera.set_state()` allows setting `x`, `y`, and `z`.
  * `Controller.add_camera()`  has new args ` include_state` and `exclude_state`.

### Fixes
* Fix volume example picking by @hmaarrfk in https://github.com/pygfx/pygfx/pull/788
* Fix `Controller.cameras` property, add a very basic test by @kushalkolar in https://github.com/pygfx/pygfx/pull/791
* Fix legacy linalg by @panxinmiao in https://github.com/pygfx/pygfx/pull/785

### Added
* Add Background.from_color by @almarklein in https://github.com/pygfx/pygfx/pull/733
* GLTF Loader by @panxinmiao in https://github.com/pygfx/pygfx/pull/732
* Additive blending mode by @tlambert03 in https://github.com/pygfx/pygfx/pull/752
* Grid object by @almarklein in https://github.com/pygfx/pygfx/pull/743
* Add SkinnedMesh by @panxinmiao in https://github.com/pygfx/pygfx/pull/715
* Add isosurface volume material by @kevinyamauchi in https://github.com/pygfx/pygfx/pull/773
* Add minip volume rendering by @kevinyamauchi in https://github.com/pygfx/pygfx/pull/784

### Added examples
* Add validation examples for blending by @almarklein in https://github.com/pygfx/pygfx/pull/734
* Integration example of pygfx with pytorch lightning and pytorch geometric by @royvelich in https://github.com/pygfx/pygfx/pull/588
* Color Conversion in PBR Example by @panxinmiao in https://github.com/pygfx/pygfx/pull/738
* Add example to show cpu usage in title by @almarklein in https://github.com/pygfx/pygfx/pull/767
* Audio visualization by @panxinmiao in https://github.com/pygfx/pygfx/pull/765
* Add text overlays to the transparency examples by @hmaarrfk in https://github.com/pygfx/pygfx/pull/782
* Tweak hello_text.py example by @almarklein in https://github.com/pygfx/pygfx/pull/779

### Internal stuff
* Refactor baseshader by @almarklein in https://github.com/pygfx/pygfx/pull/736
* Tweak wgpu abstractions by @almarklein in https://github.com/pygfx/pygfx/pull/762
* Use np.linalg.pinv instead of .inv by @almarklein in https://github.com/pygfx/pygfx/pull/766
* Fix method prefix to be consistent with other cases by @almarklein in https://github.com/pygfx/pygfx/pull/768
* Adjust to latest wgpu by @almarklein in https://github.com/pygfx/pygfx/pull/786

### Docs, tests, CI, etc.
* Suppress sphinx warning about caching gallery_conf by @almarklein in https://github.com/pygfx/pygfx/pull/735
* Update documentation to reflect that modifiers and buttons are tuples not lists by @hmaarrfk in https://github.com/pygfx/pygfx/pull/731
* Update writable flag documentation in the guide by @hmaarrfk in https://github.com/pygfx/pygfx/pull/747
* Update dev install instructions by @kevinyamauchi in https://github.com/pygfx/pygfx/pull/755
* Fix that `pytest examples` does not always work by @almarklein in https://github.com/pygfx/pygfx/pull/770
* Bump github actions to address deprecation warnings by @hmaarrfk in https://github.com/pygfx/pygfx/pull/777
* Move most builds from 3.9 to 3.12 by @almarklein in https://github.com/pygfx/pygfx/pull/793
* Rebrand by @Korijn in https://github.com/pygfx/pygfx/pull/757
* logo visibility in docs by @Korijn in https://github.com/pygfx/pygfx/pull/758
* override text color for version text in docs by @Korijn in https://github.com/pygfx/pygfx/pull/759
* Fix links by @Korijn in https://github.com/pygfx/pygfx/pull/760
* Create FUNDING.yml by @almarklein in https://github.com/pygfx/pygfx/pull/796
* Simplify texts on support a bit by @almarklein in https://github.com/pygfx/pygfx/pull/797

### New Contributors
* @royvelich made their first contribution in https://github.com/pygfx/pygfx/pull/588
* @tlambert03 made their first contribution in https://github.com/pygfx/pygfx/pull/752
* @kevinyamauchi made their first contribution in https://github.com/pygfx/pygfx/pull/755

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.2.0...v0.3.0


## [v0.2.0](https://github.com/pygfx/pygfx/releases/tag/v0.2.0) - 15-04-2024

### API changes

* World objects are not pickable by default anymore. To enable picking, set `Material.pick_write` to `True`.

### What's Changed

* Refactor wgpu renderer by @almarklein in https://github.com/pygfx/pygfx/pull/675
* Fix minigalleries in docs by @almarklein in https://github.com/pygfx/pygfx/pull/677
* Better line-render behavior for nonfintes by @almarklein in https://github.com/pygfx/pygfx/pull/680
* Documentation improvements by @almarklein in https://github.com/pygfx/pygfx/pull/679
* Point shader improvements by @almarklein in https://github.com/pygfx/pygfx/pull/681
* Make pixel_ratio follow canvas if it changes by @almarklein in https://github.com/pygfx/pygfx/pull/685
* Prevent error message about id when passing invalid args to a wobject by @almarklein in https://github.com/pygfx/pygfx/pull/688
* Fix that changing the ImageMaterial.map does not update it by @almarklein in https://github.com/pygfx/pygfx/pull/689
* Fix `look_at` by @panxinmiao in https://github.com/pygfx/pygfx/pull/691
* Fix update issues related to render_mask by @almarklein in https://github.com/pygfx/pygfx/pull/687
* PointMaterial -> PointsMaterial in error string by @hmaarrfk in https://github.com/pygfx/pygfx/pull/692
* Update image_plus_points to demostrate variable point sizes by @hmaarrfk in https://github.com/pygfx/pygfx/pull/695
* Update expected timeline for version 1.0 by @hmaarrfk in https://github.com/pygfx/pygfx/pull/703
* Updated wx rotating cube example by @cansik in https://github.com/pygfx/pygfx/pull/697
* Add point sprites by @almarklein in https://github.com/pygfx/pygfx/pull/693
* Fix wobbly points by @almarklein in https://github.com/pygfx/pygfx/pull/706
* Supports InstancedMesh Shadows by @panxinmiao in https://github.com/pygfx/pygfx/pull/705
* Allow picking of background objects by @hmaarrfk in https://github.com/pygfx/pygfx/pull/700
* Add functions to select a specific gpu by @almarklein in https://github.com/pygfx/pygfx/pull/709
* Use the render_mask to control when the background shader is displayed by @hmaarrfk in https://github.com/pygfx/pygfx/pull/711
* Make mesh dynamic example be a little more natural by @hmaarrfk in https://github.com/pygfx/pygfx/pull/710
* Improve trimesh conversion by @schlegelp in https://github.com/pygfx/pygfx/pull/707
* Fix examples with new "load_mesh" by @panxinmiao in https://github.com/pygfx/pygfx/pull/716
* Use imageiov3 to avoid warning in skybox by @hmaarrfk in https://github.com/pygfx/pygfx/pull/717
* Making material pickability controllable by @hmaarrfk in https://github.com/pygfx/pygfx/pull/704
* Refactor enums by @almarklein in https://github.com/pygfx/pygfx/pull/718
* Fix that pick_write is not dynamic by @almarklein in https://github.com/pygfx/pygfx/pull/722
* New enum for coordinate space by @almarklein in https://github.com/pygfx/pygfx/pull/723
* Add PointsMarkerMaterial by @almarklein in https://github.com/pygfx/pygfx/pull/708
* Basic new line support for left to right scripts by @hmaarrfk in https://github.com/pygfx/pygfx/pull/712
* Fix render_mask logic for points by @almarklein in https://github.com/pygfx/pygfx/pull/729
* Set the children property of the `WorldObject` class to be read-only. by @panxinmiao in https://github.com/pygfx/pygfx/pull/728
* Align and fix rendermask usage by @almarklein in https://github.com/pygfx/pygfx/pull/730


### New Contributors
* @cansik made their first contribution in https://github.com/pygfx/pygfx/pull/697
* @schlegelp made their first contribution in https://github.com/pygfx/pygfx/pull/707

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.1.18...v0.2.0


## [v0.1.18](https://github.com/pygfx/pygfx/releases/tag/v0.1.18) - 08-03-2024

### What's Changed
* Update transparency to include a background by @almarklein in https://github.com/pygfx/pygfx/pull/623
* Improve overlay rendering by @almarklein in https://github.com/pygfx/pygfx/pull/624
* Renderer always applies object.render_order by @almarklein in https://github.com/pygfx/pygfx/pull/627
* add code of conduct by @kushalkolar in https://github.com/pygfx/pygfx/pull/629
* Black by @almarklein in https://github.com/pygfx/pygfx/pull/637
* Replace pkg_resources with importlib.resources by @almarklein in https://github.com/pygfx/pygfx/pull/643
* Enable float32-filterable feature by @almarklein in https://github.com/pygfx/pygfx/pull/644
* Renderer.pixel_ratio always has a value by @almarklein in https://github.com/pygfx/pygfx/pull/638
* Update  print_wgpu_report to make use of wgpu's diagnostics system by @almarklein in https://github.com/pygfx/pygfx/pull/648
* Use wgsl saturate instrinsic when we can. by @hmaarrfk in https://github.com/pygfx/pygfx/pull/653
* Using the built-in function "smoothstep" by @panxinmiao in https://github.com/pygfx/pygfx/pull/659
* Fix a light_view and cube_qt examples by @hmaarrfk in https://github.com/pygfx/pygfx/pull/658
* Make returned arrays of gometric transforms read only by @hmaarrfk in https://github.com/pygfx/pygfx/pull/652
* Fix that colormaps were not sampled correctly by @almarklein in https://github.com/pygfx/pygfx/pull/660
* Fix off by 1 error in pieces creation for text selection by @hmaarrfk in https://github.com/pygfx/pygfx/pull/657
* Line dashing by @almarklein in https://github.com/pygfx/pygfx/pull/628
* Fix typo in show_rect docstring by @hmaarrfk in https://github.com/pygfx/pygfx/pull/662
* add zmq example of how to compute data in one process and view it in another process by @kushalkolar in https://github.com/pygfx/pygfx/pull/669
* Fix that shader is not recompiled when number of clip planes changes by @almarklein in https://github.com/pygfx/pygfx/pull/670
* Buffer and Texture test for c_contiguous by @almarklein in https://github.com/pygfx/pygfx/pull/672



**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.1.17...v0.1.18


## [v0.1.17](https://github.com/pygfx/pygfx/releases/tag/v0.1.17) - 24-11-2023

### What's Changed
* Adjust for change in wgpu (wgpu.base rename) by @almarklein in https://github.com/pygfx/pygfx/pull/613
* Adjust to new wgpu-py by @almarklein in https://github.com/pygfx/pygfx/pull/617


**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.1.16...v0.1.17


## [v0.1.16](https://github.com/pygfx/pygfx/releases/tag/v0.1.16) - 15-11-2023

### What's Changed
* Fix typos by @kianmeng in https://github.com/pygfx/pygfx/pull/603
* Fix a typo in custom shaders example by @pyetras in https://github.com/pygfx/pygfx/pull/605
* Adjust to new wgpu-py by @almarklein in https://github.com/pygfx/pygfx/pull/604

### New Contributors
* @kianmeng made their first contribution in https://github.com/pygfx/pygfx/pull/603
* @pyetras made their first contribution in https://github.com/pygfx/pygfx/pull/605

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.1.15...v0.1.16


## [v0.1.15](https://github.com/pygfx/pygfx/releases/tag/v0.1.15) - 16-10-2023


## [v0.1.14](https://github.com/pygfx/pygfx/releases/tag/v0.1.14) - 02-10-2023


## [v0.1.13](https://github.com/pygfx/pygfx/releases/tag/v0.1.13) - 12-06-2023

### Most notable changes that affect the API

* The internal `pygfx.pylinalg` has been replaced with a new (external) `pylinalg` package. Any code that used it will need to be updated to this new API.
* The `Controller` code has been refactored, and its API has changed (initialization, and how it connects to events).
* The `TextureView` class has been removed. Update your code to just use textures.

### Full list of changes
* Update _shaderlib.py to avoid a bug on Metal by @sadaszewski in https://github.com/pygfx/pygfx/pull/483
* Require pylinalg by @almarklein in https://github.com/pygfx/pygfx/pull/484
* Generate class docs in normal Sphinx way by @almarklein in https://github.com/pygfx/pygfx/pull/479
* add Texture format for uint32 by @kushalkolar in https://github.com/pygfx/pygfx/pull/473
* Cameras and controller overhaul by @almarklein in https://github.com/pygfx/pygfx/pull/457
* Stats helper by @Korijn in https://github.com/pygfx/pygfx/pull/486
* Add look-at example by @Korijn in https://github.com/pygfx/pygfx/pull/488
* Fix shadow camera cull mode by @Korijn in https://github.com/pygfx/pygfx/pull/477
* Fix skybox by @panxinmiao in https://github.com/pygfx/pygfx/pull/466
* Enable getting full uniform_type from classes. by @almarklein in https://github.com/pygfx/pygfx/pull/497
* Add frustum property to perspective camera by @FirefoxMetzger in https://github.com/pygfx/pygfx/pull/498
* Remove TextureView from public api by @almarklein in https://github.com/pygfx/pygfx/pull/493
* Mipmap with proper smoothing by @almarklein in https://github.com/pygfx/pygfx/pull/501
* Make controllers configurable and apply damping by @almarklein in https://github.com/pygfx/pygfx/pull/490
* Add trackball controller by @almarklein in https://github.com/pygfx/pygfx/pull/503
* Tweak reactive example by @almarklein in https://github.com/pygfx/pygfx/pull/502
* Add fly controller by @almarklein in https://github.com/pygfx/pygfx/pull/504
* More flexibly detect sponza data. Als show fps in sponza scene by @almarklein in https://github.com/pygfx/pygfx/pull/505
* Skybox validation by @almarklein in https://github.com/pygfx/pygfx/pull/507
* more informative exception message for `Buffer.update_range` valid types by @kushalkolar in https://github.com/pygfx/pygfx/pull/511
* Fix that renderer-printed fps was multiplied by number of renders per draw by @almarklein in https://github.com/pygfx/pygfx/pull/514
* More efficient Buffer and Texture updates by @almarklein in https://github.com/pygfx/pygfx/pull/508
* New cache class, apply to mipmaps by @almarklein in https://github.com/pygfx/pygfx/pull/516
* Tweak resource update code a bit by @almarklein in https://github.com/pygfx/pygfx/pull/517
* Refactor pygfx using pylinalg by @FirefoxMetzger in https://github.com/pygfx/pygfx/pull/482
* Apply pylinalg rename by @Korijn in https://github.com/pygfx/pygfx/pull/524
* Remove pyinstaller workaround for freetype by @Korijn in https://github.com/pygfx/pygfx/pull/526
* Caching in _pipeline.py by @almarklein in https://github.com/pygfx/pygfx/pull/525
* Fix ups by @almarklein in https://github.com/pygfx/pygfx/pull/527
* Shadow caching and improvements  by @almarklein in https://github.com/pygfx/pygfx/pull/530
* Implement cashing for renderer, blender, flusher by @almarklein in https://github.com/pygfx/pygfx/pull/533
* Final caching tweaks by @almarklein in https://github.com/pygfx/pygfx/pull/535
* Refactor usage of global device by @almarklein in https://github.com/pygfx/pygfx/pull/536
* add example of how to map screen to world by @kushalkolar in https://github.com/pygfx/pygfx/pull/512
* Add more extensive envmap supports by @panxinmiao in https://github.com/pygfx/pygfx/pull/406
* exclude nans in `Geometry.bounding_box()` by @kushalkolar in https://github.com/pygfx/pygfx/pull/540
* Show line caps on line segments separated by nan by @almarklein in https://github.com/pygfx/pygfx/pull/541
* Option to make a solid torus knot by @almarklein in https://github.com/pygfx/pygfx/pull/542
* Add Mobius strip (and some more) by @almarklein in https://github.com/pygfx/pygfx/pull/543
* Fix that the polyhedron indices array was 3 times too long by @almarklein in https://github.com/pygfx/pygfx/pull/544
* Adjust rtd links to point to stable by @almarklein in https://github.com/pygfx/pygfx/pull/546
* Clear up some expectations in the readme by @almarklein in https://github.com/pygfx/pygfx/pull/547
* Bump version by @almarklein in https://github.com/pygfx/pygfx/pull/548

### New Contributors
* @sadaszewski made their first contribution in https://github.com/pygfx/pygfx/pull/483

**Full Changelog**: https://github.com/pygfx/pygfx/compare/v0.1.12...v0.1.13


## [v0.1.12](https://github.com/pygfx/pygfx/releases/tag/v0.1.12) - 23-02-2023


## [v0.1.11](https://github.com/pygfx/pygfx/releases/tag/v0.1.11) - 15-02-2023


## [v0.1.10](https://github.com/pygfx/pygfx/releases/tag/v0.1.10) - 12-01-2023


## [v0.1.9](https://github.com/pygfx/pygfx/releases/tag/v0.1.9) - 20-04-2022


## [v0.1.8](https://github.com/pygfx/pygfx/releases/tag/v0.1.8) - 12-04-2022


## [v0.1.7](https://github.com/pygfx/pygfx/releases/tag/v0.1.7) - 07-03-2022


## [v0.1.6](https://github.com/pygfx/pygfx/releases/tag/v0.1.6) - 04-02-2022


## [v0.1.5](https://github.com/pygfx/pygfx/releases/tag/v0.1.5) - 05-11-2021


## [v0.1.4](https://github.com/pygfx/pygfx/releases/tag/v0.1.4) - 11-10-2021
