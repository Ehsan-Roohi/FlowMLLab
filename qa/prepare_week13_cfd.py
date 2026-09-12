"""Generate independent OpenFOAM v2406 laminar cavity reference candidates.

Outputs are research candidates, not validated reference data. No dependencies.
"""
import argparse
import json
from pathlib import Path


CASES = [(r, 1.0) for r in (100, 400, 1000)] + [
    (1000, d) for d in (1.1, 1.15, 1.2, 1.25, 2.2, 2.3, 2.4, 3.2)
] + [(r, 5.0) for r in (100, 500, 1000)] + [(r, 7.0) for r in (500, 1000)] + [
    (500, 2.2)
]
GRIDS = (80, 120, 180)


def dictionary(name, body, cls="dictionary"):
    return (f"FoamFile\n{{version 2.0; format ascii; class {cls}; object {name};}}\n"
            + body + "\n")


def files(re, depth, nx, iterations, ranks):
    ny = round(nx * depth)
    # Symmetric edge grading: same mesh family on all refinement levels.
    grade = "((0.5 0.5 4) (0.5 0.5 0.25))"
    mesh = f"""scale 1;
vertices ((0 0 0) (1 0 0) (1 {depth} 0) (0 {depth} 0)
          (0 0 0.01) (1 0 0.01) (1 {depth} 0.01) (0 {depth} 0.01));
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1)
        simpleGrading ({grade} {grade} 1));
edges ();
boundary (
 lid {{type wall; faces ((3 7 6 2));}}
 walls {{type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4));}}
 frontAndBack {{type empty; faces ((0 3 2 1) (4 5 6 7));}}
);
mergePatchPairs ();
"""
    return {
        "system/blockMeshDict": dictionary("blockMeshDict", mesh),
        "0/U": dictionary("U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField {
 lid {type fixedValue; value uniform (1 0 0);}
 walls {type fixedValue; value uniform (0 0 0);}
 frontAndBack {type empty;}
}""", "volVectorField"),
        "0/p": dictionary("p", """dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField {
 lid {type zeroGradient;}
 walls {type zeroGradient;}
 frontAndBack {type empty;}
}""", "volScalarField"),
        "constant/transportProperties": dictionary("transportProperties",
            f"transportModel Newtonian;\nnu [0 2 -1 0 0 0 0] {1/re:.17g};"),
        "constant/turbulenceProperties": dictionary("turbulenceProperties", "simulationType laminar;"),
        "system/controlDict": dictionary("controlDict", f"""application simpleFoam;
startFrom latestTime; startTime 0; stopAt endTime; endTime {iterations}; deltaT 1;
writeControl timeStep; writeInterval {min(500, iterations)}; purgeWrite 3;
writeFormat binary; writePrecision 16; writeCompression off;
timeFormat general; timePrecision 12; runTimeModifiable true;
"""),
        "system/fvSchemes": dictionary("fvSchemes", """ddtSchemes {default steadyState;}
gradSchemes {default Gauss linear;}
divSchemes {default none; div(phi,U) Gauss linear; div((nuEff*dev2(T(grad(U))))) Gauss linear;}
laplacianSchemes {default Gauss linear corrected;}
interpolationSchemes {default linear;}
snGradSchemes {default corrected;}
wallDist {method meshWave;}
"""),
        "system/fvSolution": dictionary("fvSolution", """solvers {
 p {solver GAMG; tolerance 1e-13; relTol 0.01; smoother GaussSeidel;}
 U {solver smoothSolver; smoother symGaussSeidel; tolerance 1e-13; relTol 0.01;}
}
SIMPLE {nNonOrthogonalCorrectors 0; consistent yes; pRefCell 0; pRefValue 0;}
relaxationFactors {fields {p 0.2;} equations {U 0.5;}}
"""),
        "system/decomposeParDict": dictionary("decomposeParDict",
            f"numberOfSubdomains {ranks}; method scotch;"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ranks", type=int, default=8)
    ap.add_argument("--sensitivity", action="store_true")
    args = ap.parse_args()
    if not 0 <= args.task < len(CASES) * len(GRIDS):
        ap.error(f"task must be 0..{len(CASES) * len(GRIDS) - 1}")
    re, depth = CASES[args.task // len(GRIDS)]
    nx = 20 if args.smoke else GRIDS[args.task % len(GRIDS)]
    iterations = 20 if args.smoke else 20000
    variant = "baseline"
    if args.sensitivity:
        if not 0 <= args.task < 6:
            ap.error("sensitivity task must be 0..5")
        re, depth = (100, 500)[args.task % 2], 5.0
        variant = ("refine270", "leastSquares", "linearUpwind")[args.task // 2]
        nx = 20 if args.smoke else (270 if variant == "refine270" else 180)
        iterations = 20 if args.smoke else 120000
    spec = dict(format_version=2, re=re, depth_over_width=depth, nx=nx, ny=round(nx*depth),
                ranks=args.ranks, iterations=iterations, lid="uniform/classical",
                solver="OpenFOAM-v2406/simpleFoam/laminar", status="unvalidated-candidate",
                reference="https://doi.org/10.1016/j.compfluid.2005.08.006")
    if args.sensitivity:
        spec["sensitivity_variant"] = variant
    path = args.output.resolve()
    marker = path / "case-spec.json"
    if path.exists():
        if not marker.exists() or json.loads(marker.read_text()) != spec:
            raise RuntimeError("Existing directory is not this exact case; refusing overwrite")
        print(path)
        return
    path.mkdir(parents=True)
    generated = files(re, depth, nx, iterations, args.ranks)
    if variant == "leastSquares":
        generated["system/fvSchemes"] = generated["system/fvSchemes"].replace(
            "gradSchemes {default Gauss linear;}", "gradSchemes {default leastSquares;}")
    elif variant == "linearUpwind":
        generated["system/fvSchemes"] = generated["system/fvSchemes"].replace(
            "div(phi,U) Gauss linear;", "div(phi,U) Gauss linearUpwind grad(U);")
    for name, content in generated.items():
        dest = path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(content, encoding="ascii", newline="\n")
    marker.write_text(json.dumps(spec, indent=2) + "\n", encoding="ascii")
    print(path)


if __name__ == "__main__":
    main()
