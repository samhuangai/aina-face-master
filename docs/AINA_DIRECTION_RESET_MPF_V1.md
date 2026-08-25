# AINA Identity Direction Reset — MPF v1

## Decision

The Rain-based direction is rejected as the production identity base. The actual Blender renders prove that Rain's chibi cranial, eye, jaw and facial-surface priors remain dominant after repeated topology-preserving fitting. No further Rain v8/v9 identity sculpting is permitted.

## New production direction

Build a dedicated AINA identity head from an adult parametric human base generated with MakeHuman/MPFB, then fit and sculpt it directly from the already-approved AINA front, three-quarter and profile references.

This is not another generic-character repaint. The workflow is:

1. Generate a young adult East-Asian female basemesh with semantic macro and face targets.
2. Solve camera and facial target weights against approved front / 3Q / side references.
3. Bake the target stack into a dedicated AINA neutral head.
4. Apply one constrained custom corrective sculpt for eyelids, nose tip/ala, lips, apple cheeks, jaw and chin.
5. Produce naked-head Beauty and Clay front / 20° / 45° / left / right profile QA.
6. Lock identity only after direct visual approval.
7. After identity lock, build 52 ARKit controls, 18 VRM presets, Humanoid, LookAt, SpringBone and clean reimport QA.

## Hard gates

The project must not proceed to hair, body or VRM while any of the following are false:

- The neutral front reads as the approved AINA.
- The neutral 3Q reads as the same person.
- The profile has the approved forehead / nose / lips / chin relationship.
- Eye size and spacing are adult semi-realistic rather than chibi.
- Nose and lips are integrated into the facial surface, not separate patches.
- The lower third and V jaw remain stable across views.

## Frozen / rejected paths

- Rain identity fitting: rejected for visual identity.
- Rain VRM v7: rejected as a production release; technical mapping failure and visual failure remain unresolved.
- More sparse-landmark-only local warps: prohibited.
- Declaring identity lock from automated geometry metrics alone: prohibited.

## Delivery target

- Blender 4.5 LTS editable master.
- Dedicated AINA neutral head topology suitable for deformation.
- 52 non-zero ARKit controls.
- VRM 1.0 with 18 presets, Humanoid, LookAt and SpringBone.
- Exact-byte clean reimport and actual-render QA.
