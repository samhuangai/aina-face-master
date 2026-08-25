# AINA Direction Reset — Custom Identity Head

## Decision

The Rain facial topology is no longer the primary AINA head direction.

Repeated actual Blender renders show that Rain remains structurally chibi after bounded fitting: oversized eye sockets and sclera, short rounded mid/lower face, broad cranium, small nose volume and weak adult facial planes. Continuing to push that topology produces either the same chibi identity or collapsed nose/lip/jaw geometry.

Rain is retained only as an optional body, armature, clothing and secondary-animation source. Its face is not used as the identity master.

## New primary direction

AINA now uses a custom adult identity head built from the approved front, three-quarter and side references:

1. Reconstruct a dense neutral target with 3DDFA/BFM from all approved views.
2. Reconstruct an expression-capable FaceVerse neutral head from the same approved views.
3. Non-rigidly wrap the FaceVerse production topology to the dense target with trimmed similarity ICP, distance confidence, Laplacian regularization and bounded displacement.
4. Preserve FaceVerse vertex order and save the neutral displacement field so it can be applied identically to every expression target.
5. Validate the real custom head in front, 3Q, 45-degree and profile clay views before adding hair, body or VRM packaging.
6. After visual approval, transfer 52 facial controls, attach the head to an adult body/armature, then build VRM 1.0 and perform exact-byte clean reimport.

## Hard stop rules

- No more Rain face versions.
- No more VRM packaging before the neutral custom head passes visual review.
- No automatic `identity_lock` from landmark metrics alone.
- No replacement effect-art image is accepted as model evidence.
- A topology is rejected immediately when actual renders show a structural mismatch that requires destructive deformation.

## Current state

```text
primary_face_direction: CUSTOM_FACEVERSE_DENSE_GRAFT
rain_face_direction: REJECTED
rain_body_and_rig: OPTIONAL_REUSE
identity_lock: false
visual_identity_lock: false
production_release: false
```
