# AINA Face Master

Production repository for the AINA neutral head identity reconstruction pipeline.

## Goal
Build one real 3D neutral head mesh whose facial identity matches the approved AINA effect-art reference. Hair, clothing, expressions, rigging and VRM are intentionally blocked until the clay head passes identity QA.

## Pipeline
1. Use Google GNM v3 HEAD as the anatomical production base.
2. Detect 68 facial landmarks from the approved AINA neutral reference.
3. Fit GNM head identity coefficients under a neutral-expression constraint.
4. Export OBJ/PLY/GLB plus identity coefficients.
5. Render front, ±45° and ±90° clay views.
6. Produce reference-overlay QA and numerical fit report.

The build is executed by GitHub Actions and publishes its outputs as the `aina-face-master-build` artifact.
