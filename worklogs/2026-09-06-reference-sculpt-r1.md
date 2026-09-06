# AINA Reference Sculpt R1 — working head, not final character

Date: 2026-09-06

The active modeling session produced a real editable Blender head and a static GLB. This is not the previously claimed V21 release and is not an approved AINA identity.

## Actual scope

- Source: candidate 14 from the MakeHuman-based custom-sculpt artifact; front reference is the supplied AINA concept portrait.
- Symmetric front anatomical control field, explicit eyelid aperture refinement, local brow/nose/lip/chin-neck shaping.
- Independent eyeballs, curved irises with an original procedural iris texture, editable eyelash curves, neutral oral helper surfaces.
- Head control cage: 4,689 vertices / 4,652 polygons.
- Source keys: Basis and Previous_Candidate14. The latter is only a previous neutral head comparison, NOT a viseme or expression.
- No armature, no production expression system, no full body, no final hairstyle or costume, no VRM.
- Local eyebrow/lip/eyelid pigment is reference-derived and front-surface gated. Texture-free clay renders are provided separately; color resemblance is not treated as a geometry pass.

## Verification actually performed

Blender 4.5.12 LTS reopened the source. The GLB imported in a new scene: 80 mesh objects, 32,325 imported vertices, 59,136 triangles, 6 materials, 2 embedded images. Source referenced images were packed. Source mesh coordinates were finite and the checked mesh surfaces had no zero-area faces. Open bust/iris/helper edges are recorded and are not claimed to form a closed printable or animation-ready body.

## Delivery location

The binary files were delivered as conversation artifacts, NOT committed as repository binary assets. These names/hashes identify that exact delivery; they are not repository download links.

| Artifact | Bytes | SHA256 |
|---|---:|---|
| AINA_REFERENCE_SCULPT_R1.blend | 2243818 | b4095be29847311d3bb23938683d056ed88a1bb20de79bf6b1cbc0320306204b |
| AINA_REFERENCE_SCULPT_R1.glb | 2582936 | f515734cf66c663669df75412b677dbf0bb3d628fa1ca8439c772fc8c0f23f4c |
| AINA_REFERENCE_SCULPT_R1_SOURCE.zip | 11476829 | ff4f1e4e5428823d6e7d2c6c6f8e3e8c942d875e973b037112ba0297fd290e3e |

The ZIP contains the editable head, GLB, geometry OBJ, supplied references, real rendered front/30-degree/profile views, texture-free clay views, package checksums, MakeHuman asset license text, and QA reports.

## Gates

identity_lock = false
user_identity_approval = false
final_character = false
rigged = false
vrm_exported = false

Brow/eye expression, nasal-labial form and profile volumes still differ from the concept. Experimental hair/costume did not meet the reference and were excluded from the head delivery. A successful file export is not a successful character likeness test.
