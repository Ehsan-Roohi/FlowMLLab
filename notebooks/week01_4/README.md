# Week 1.4 - Spectral and discontinuous Galerkin methods on CPU

[Course home](../../README.md) · [Weekly guide](../../weeks/week01_4/README.md) · [Lecture PDF](../../lectures/week01_4_spectral_dg.pdf)

**Prerequisite:** the Week 1 cavity foundations and [Week 1.2 pressure-velocity coupling](../week01_2/README.md). The optimizer unit numbered 1.3 is a separate branch and is not required here.

Open [W1_4_Spectral_DG_CPU.ipynb](W1_4_Spectral_DG_CPU.ipynb). All solver definitions are inside the notebook, so it can run without a repository clone, external CFD data, Nektar++, ExaDG or a GPU. The local CPU execution record is [retained here](../../qa/week01_4_execution.json); it does not certify a hosted Colab run.

## Setup

Use Python 3.10-3.13 and install `numpy`, `scipy`, `matplotlib`, `pandas` and a Jupyter interface. The course environment also works. In Colab, upload the notebook and select a CPU runtime. If a package is missing, run `%pip install numpy scipy matplotlib pandas`, restart the kernel and run from the top.

Run All recomputes interpolation, manufactured Poisson, periodic DG transport, and actual steady incompressible cavities at **Re=100 and 400**. No CFD field is silently loaded from a reference cache. The default run saves eight tables, eleven figures (PNG and vector PDF), six compact cavity field files, and `execution.json` to `week01_4_output`. Set `FLOWMLLAB_W14_OUTPUT` to choose another output directory. Start with the supplied controls; choosing only Re=100 gives a shorter first pass. The recorded complete CPU run takes a few minutes, including figures; this is not a portable runtime guarantee.

## What to study

| Notebook sections | Topic | Student control | Evidence |
| --- | --- | --- | --- |
| 1-2 | Global Chebyshev interpolation and differentiation | Polynomial degree and smoothness | Table A; field and derivative errors |
| 3-4 | Tensor-product Poisson versus finite differences | Resolution and accuracy target | Table P; pressure contours/profiles, residual, repeated timing |
| 5-6 | Legendre DG from the weak form | Cell count, local degree, flux and time step | Conservative residual and SSPRK3 algorithm |
| 7-9 | Transport, discontinuities and refinement | Central/upwind flux, h/p budget, CFL and speed sign | Tables D/F/T/H; signed errors, mass, energy, work |
| 10 | Actual Chebyshev Navier-Stokes cavity | Reynolds number, N=8-32, Newton tolerance/budget, standard or smooth lid | Computed velocity, pressure, vorticity, streamlines, Ghia profiles and independent continuity diagnostics |
| 11-12 | Nine investigations and reproducibility | One controlled change per experiment | Executed notebook, tables/fields and an engineering recommendation |

Keep the problems distinct: manufactured Poisson uses prescribed zero pressure walls, whereas the actual cavity solves for pressure with a centre gauge and prescribed wall velocities. Its solver uses mixed Chebyshev velocity/pressure spaces and a coupled steady Newton iteration. The DG example remains periodic scalar transport; the cavity is not labelled as DG. Small collocation residuals and off-grid continuity errors are reported separately. Standard-lid Ghia comparisons retain all printed entries and also disclose the predeclared Re=400 screening sensitivity. A smooth lid changes the benchmark, so Ghia scoring is disabled for it.

## Teaching and reproduction source

- [Tested numerical functions](../../flowmllab/high_order.py). Their exact definitions are embedded in the notebook during authoring.
- [Polynomial, manufactured-PDE, conservation, energy and refinement checks](../../tests/test_high_order.py).
- [Actual spectral cavity solver](../../flowmllab/high_order_cavity.py) and [independent Navier-Stokes/Jacobian/wall/refinement tests](../../tests/test_high_order_cavity.py).
- [Notebook authoring script](../../qa/build_week01_4_notebook.py).
- [Fresh local execution helper](../../qa/execute_week01_4.py).
- [Lecture source and builder](../../qa/build_week01_4_lecture.py), using the executed notebook's tables and figures; it requires the course's licensed Times New Roman typography setup.
- [Retained results](../../results/week01_4_spectral_dg/README.md).

The original Week 1.2 notebook, lecture and exercises remain available at their existing addresses.
