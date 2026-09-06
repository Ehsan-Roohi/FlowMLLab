# Reserved cylinder teaching test: Re = 115

Generated independently with the existing D2Q9 TRT/Bouzidi solver and the v1
240 x 96 mesh, D=12, U=0.05, seed=690, 22000 timesteps. Geometry and the full
force history are preserved. For compact distribution, every fifth field
snapshot is retained: 57 fields at 125-timestep intervals, selected by index
without screening the solution. The original solver metadata records the
25-timestep sampling; export subsampling is explicit in the metadata/manifest.

No surrogate fitting, model selection or error scoring has used this case in
this change. A public checksum is a provenance seal, not an access restriction.
Freeze the model, preprocessing and assessment protocol before first scoring;
then log first use and retire blind-test status after feedback. Integrity checks
passed; this is not a grid-independent or statistically certified DNS benchmark.

Reproduce from the repository root with:
`python qa/generate_sealed_cylinder.py --output tmp/reserved-new`
(the destination must not exist). Compare numerical arrays if package versions
change; ZIP timestamps can affect archive hashes. Verify the distributed bytes
against `manifest.json` before use. Do not use Strouhal metadata to tune a model.

The archive is distributed in two binary parts. Run
`python data/cylinder_reserved_re115/assemble.py` from the repository root to
reassemble and verify it without loading or scoring the fields.
