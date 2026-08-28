# Training-Free Face Geometry Direction

This experiment drops seed/trajectory selection as a core method. Seed picking can improve reported examples without changing the generator, so it is not a clean identity-preservation mechanism.

## First runnable prototype

The first practical method is identity-specific face-control retargeting:

1. Detect the reference face with InsightFace.
2. Use the reference person's `landmark_3d_68` as a lightweight 3D identity proxy.
3. For each driving frame, estimate the driving head pose.
4. Re-pose the reference 3D landmarks to that driving head pose.
5. Replace only the face landmarks in the DWPose control video.
6. Keep body and hands from the original driving pose.
7. Run frozen CoInteract P2V on the new pose video.

This is not full DECA/FLAME yet, but it tests the central hypothesis: full-body face drift partly comes from generic or driver-shaped face landmarks, especially under yaw/profile changes.

## Implemented experiment package

- `scripts/id_face_retarget.py`
- `configs/p2v_geometry_face_control_20260823.json`
- `scripts/run_p2v_geometry_face_control_20260823.sh`
- `scripts/eval_p2v_geometry_face_control_20260823.sh`
- `scripts/make_pose_face_retarget_sheet_20260823.py`

Variants:

- Original DWPose baseline.
- Identity-retargeted DWPose face baseline.
- Identity-retargeted face plus mild reference K/V guidance (`head_attn`, scale `0.05`).
- Identity-retargeted face plus latent-blur/SAMG/APG velocity guidance.

## Next research upgrades

- Safer landmark retargeting: partial blends (`0.5`, `0.75`) instead of replacing the full DWPose face.
- DWPose-compatible face mesh overlay: draw a richer identity face wireframe into the control image without changing channels.
- Replace InsightFace 3D-68 with DECA/FLAME identity shape.
- Render dense identity-specific face proxy controls: mesh lines, normals, depth, or visibility.
- Build a synthetic multi-view face bank from the fitted 3D identity.
- Use view-conditioned reference interpolation rather than feeding all views at once.
- If training-free methods hit a ceiling, train a small frozen-CoInteract face adapter with 2D identity features plus 3D geometry features.
