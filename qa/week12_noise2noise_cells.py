"""Reviewable source of the real-data section inserted into the Week 12 lab."""
import textwrap
import nbformat as nbf


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def cells():
    return [md(r'''
    ## 7. Train a real DSMC denoiser, not a figure viewer
    **Question:** can independent noisy observations teach a patch MLP to reduce
    sampling noise without using a high-budget reference during training?
    These qx/qy fields come from Roohi's existing JCP2 cavity archive associated
    with [arXiv:2609.01637](https://doi.org/10.48550/arXiv.2609.01637): Kn=0.085,
    lid speed 350 m/s, eight three-block observations on the same 100x100 grid.
    No archived neural predictions are inputs or targets of this exercise.

    This is **same-condition denoising**, not new-condition generalization or
    reproduction of the paper's MambaIR/cylinder estimator. The archive was
    already inspected before this teaching split; do not call this a newly blind trial.
    Plan 20-30 additional classroom minutes; actual CPU training time is printed.

    Method inspiration: Lehtinen et al.,
    [Noise2Noise (ICML 2018)](https://arxiv.org/abs/1803.04189).
    The implementation here is original; no NVlabs code is incorporated.
    For paired observations $Y=f+\epsilon$ and $Y'=f+\epsilon'$, squared-loss
    training against $Y'$ has the same population optimum as training against
    $f$ if $E[\epsilon'\mid Y,f]=0$. Independent, conditionally zero-mean target
    noise is sufficient. Distinct filenames alone do not establish this condition.
    Finite-sample central moments can be biased; this lab does not prove their
    unbiasedness or recover the exact kinetic solution.
    '''),code('''
    import hashlib, json, time
    import flowmllab.noise2noise as nn
    data_dir = ROOT / 'data/week12_noise2noise'
    n2n_manifest = json.loads((data_dir/'manifest.json').read_text())
    for name, expected in n2n_manifest['files'].items():
        assert hashlib.sha256((data_dir/name).read_bytes()).hexdigest() == expected
    with np.load(data_dir/'observations.npz', allow_pickle=False) as data_file:
        real_seeds = data_file['seeds'].copy()
        real_observations = data_file['raw3'].astype(np.float64)
    groups = nn.split_indices(real_seeds, n2n_manifest['split'])
    display(pd.DataFrame({name: pd.Series(real_seeds[ids]) for name,ids in groups.items()}))
    print(real_observations.shape, '=(seeds, components, rows, columns)')
    print(n2n_manifest['independence'])
    '''),md('''
    ### 7.1 Freeze whole-seed splits and build independent targets
    Training: seeds 26082101-104; validation: 105-106; evaluation: 107-108.
    For each training input, average only the OTHER three training seeds to form
    its target. Thus the target uses nine observation blocks, disjoint from the
    input's three blocks according to the source archive. The four input/target
    pairs reuse training seeds; they are not four independent paired experiments.
    Validation targets are the other validation seed, never training or test data.

    A 5x5 two-component patch supplies 50 inputs to a fixed 64x32 tanh MLP.
    No coordinates are supplied. Reflection padding is a numerical assumption,
    not a verified physical boundary condition; report edge-band errors separately.
    Scales are fitted on training data only. Training uses 3,000 locations per seed,
    fixed initialization 12, no random-pixel validation split and at most 160 epochs.
    Spatial samples and overlapping patches are not independent statistical units.
    '''),code('''
    train_fields = real_observations[groups['train']]
    validation_fields = real_observations[groups['validation']]
    noisy_targets = nn.independent_targets(train_fields)
    np.testing.assert_allclose(noisy_targets[0], train_fields[1:].mean(axis=0))
    fit_started = time.perf_counter()
    fitted_n2n = nn.fit_experiment(train_fields, validation_fields)
    print('CPU fit seconds:', round(time.perf_counter()-fit_started, 2))
    print('MLP iterations:', fitted_n2n['mlp'].model[-1].n_iter_)
    print('Last training loss:', fitted_n2n['mlp'].model[-1].loss_)
    print('Gaussian width selected without reference:', fitted_n2n['gaussian_width'])
    display(pd.DataFrame({'sigma':list(fitted_n2n['gaussian_validation']),
                          'paired noisy validation MSE':list(fitted_n2n['gaussian_validation'].values())}))
    '''),md(r'''
    ### 7.2 Baselines and sampling budgets
    Compare Raw(3), validation-selected Gaussian filtering, a development-fitted
    DCT filter, the training mean, the fresh MLP, and that MLP after restoring
    the observation mean. The last two are separately reported, not selected by
    test error. Mean restoration preserves a measurement, not exact truth.

    The spectral baseline estimates $N_k$ from between-seed coefficient variance
    and $S_k=\max(\bar z_k^2-N_k/4,0)$ from the four training fields, then uses
    $G_k=S_k/(S_k+N_k)$, with $G_{00}=1$. Four fields give a noisy estimate of
    spectral power; no benefit is assumed. There is no spectral tuning on the reference.
    The training-mean baseline exposes the advantage of knowing the same underlying
    flow: it uses 12 development blocks and no current observation. The MLP and
    spectral method also have this development information. Raw(10) is an additional
    larger-budget comparator, NOT a matched-three-block method.

    Now open the independent 260-block reference and Raw(10) arrays for evaluation.
    The archive documents disjoint reference seeds and Raw(3)/Raw(10) observations.
    We preserve this provenance, but have not independently reconstructed RNG streams.
    '''),code('''
    with np.load(data_dir/'evaluation_only.npz', allow_pickle=False) as eval_file:
        real_reference = eval_file['reference'].copy()
        real_raw10 = eval_file['raw10'].copy()
    n2n_rows = []
    n2n_predictions = {}
    for i in groups['test']:
        estimates_real = nn.predict_experiment(fitted_n2n, real_observations[i])
        estimates_real['Raw(10), larger budget'] = real_raw10[i]
        n2n_predictions[int(real_seeds[i])] = estimates_real
        for method, estimate in estimates_real.items():
            for component, scores_real in zip(['qx','qy'], nn.audit_errors(
                    estimate, real_reference, real_observations[i])):
                n2n_rows.append({'seed':int(real_seeds[i]), 'field':component,
                                 'method':method, **scores_real})
        projected = estimates_real['MLP + observed mean']
        np.testing.assert_allclose(projected.mean(axis=(-2,-1)),
            real_observations[i].mean(axis=(-2,-1)), rtol=0, atol=1e-12)
    n2n_metrics = pd.DataFrame(n2n_rows)
    display(n2n_metrics)
    display(100*n2n_metrics.groupby(['field','method'])[
        ['reference_nrmse','gradient_nrmse','edge_nrmse']].mean())
    '''),md('''
    ### 7.3 Read the fields and failures
    Reference NRMSE uses equal-cell relative L2 error. Gradient errors use the
    same array-index finite differences on prediction and reference; their
    reference is noisy too. The edge band is the outer 10% of array rows/columns,
    not a separately verified physical boundary layer. Mean changes remain in
    archive units. Only two evaluation seeds are available: no significance,
    calibrated uncertainty or speedup claim is justified.

    The first held-out seed is displayed by predeclared order, not by its error.
    Shared color limits cover all values; nearest-pixel display retains native
    100x100 resolution. The full table retains both seeds, both components and
    all baselines, even when a simpler estimator wins.
    '''),code('''
    first_test_index = groups['test'][0]
    first_test_seed = int(real_seeds[first_test_index])
    first_estimates = n2n_predictions[first_test_seed]
    nn.contour_figure(real_observations[first_test_index],
        first_estimates['Noise2Noise MLP'], real_reference,
        component='qy', seed=first_test_seed)
    plt.show()
    nn.diagnostic_figure(first_estimates, real_reference, n2n_rows)
    plt.show()
    '''),md('''
    ### 7.4 Assignment and qualification gate
    1. Explain why using the input seed in its target rewards noise copying.
    2. Compare the training mean and MLP: does this dataset demonstrate general
       denoising, or primarily repeated observation of one fixed field?
    3. Report whether restoring the observed mean improves or worsens each field.
       Preserve both outcomes and explain why mean preservation need not lower MSE.
    4. Explain why correlated overlapping blocks and biased heat-flux estimators
       violate the simplest Noise2Noise assumptions.
    5. Before extending to another Kn or lid speed, declare entire-condition
       holdouts and acquire qualified independent realizations. Do not relabel
       these already-inspected seeds as a fresh blind benchmark.

    **Deliverable:** seed protocol, loss history summary, all 28 component/method/seed
    scores, the two-component profiles, and one paragraph about the strongest
    baseline and remaining bias. Do not force neural superiority.
    This section trains a new small teaching model from real data on every Run All;
    it writes no files. It does not reproduce Roohi's research neural architecture.
    ''')]
