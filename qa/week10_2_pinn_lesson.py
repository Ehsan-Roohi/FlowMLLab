"""Additive cells for the CPU, from-scratch characteristic BGK PINN lesson."""


def append_pinn_cells(md, code):
    md('w10-2-fresh-intro', r'''
        ## 10. Train a kinetic PINN from zero on CPU

        This section **trains a new network from random weights**. It does not load
        a research checkpoint. It then runs an independent finite-volume BGK DVM.
        The DVM is used only after fitting. This small teaching experiment is not
        a reproduction of the paper's full training budget or its Shakhov model.

        The physical problem is fixed: $x\in[-0.5,0.5]$, $t_f=0.02$, $\tau=1$,
        $R_g=1$. Let $H(x)=[1+\tanh(20x)]/2$. Initially
        $\rho_0=1-0.875H$, $u_0=0$, $T_0=1-0.2H$; the initial distribution is
        **one local Maxwellian of these macros**, not a mixture of reservoirs.
        Incoming molecules at the left/right boundaries have reservoir states
        $(\rho,u,T)=(1,0,1)$ and $(0.125,0,0.8)$. Outgoing molecules leave freely.

        A tanh network maps $(t/t_f,2x)$ to three raw values $(a_\rho,a_u,a_T)$.
        Its source macros are $\rho_\theta=\rho_0\exp[0.5(t/t_f)a_\rho]$,
        $u_\theta=(t/t_f)a_u$, and $T_\theta=T_0\exp[0.5(t/t_f)a_T]$.
        Positivity and the initial source state are therefore exact. These source
        macros parameterize Maxwellians along characteristics; they are **not**
        substituted for the plotted distribution moments.

        **Why include the untrained prediction?** The collision fraction is at
        most $1-\exp(-t_f/\tau)\simeq0.0198$. Exact transport of the known initial
        and incoming distributions already accounts for much of this short-time
        solution. A good-looking initial curve is not evidence of learned physics.
        Compare initial and trained errors, including signed nonequilibrium fields.
    ''')
    code('w10-2-fresh-dependencies', '''
        import importlib.util
        if importlib.util.find_spec('torch') is None:
            subprocess.run([sys.executable,'-m','pip','install','torch>=2.5,<3',
                            '--index-url','https://download.pytorch.org/whl/cpu'],check=True)
        import torch
        from dataclasses import replace, asdict
        from flowmllab import kinetic_pinn as kp, kinetic_dvm as kd
        print('PyTorch',torch.__version__,'; this lesson explicitly uses CPU and float64.')
    ''')
    md('w10-2-fresh-method', r'''
        ### The equation actually optimized

        Integrate out the two transverse velocities analytically. At longitudinal
        velocity $v$, retain $g=\int f\,dv_y\,dv_z$,
        $h=\int(v_y^2+v_z^2)f\,dv_y\,dv_z$ and
        $k=\int(v_y^2+v_z^2)^2f\,dv_y\,dv_z$.
        Their equilibrium values are $(g_M,h_M,k_M)=(M_x,2TM_x,8T^2M_x)$,
        where $M_x=\rho(2\pi T)^{-1/2}\exp[-(v-u)^2/(2T)]$.

        For any one of $Y=g,h,k$, trace back for duration $\ell$ to the initial
        line or first incoming boundary. With Gauss-Legendre nodes $z_j$ and
        weights $w_j$ on $[0,1]$, reconstruct
        $$Y_\theta(t,x,v)=e^{-\ell/\tau}Y_b+
        \frac{\ell}{\tau}\sum_j w_j e^{-\ell z_j/\tau}
        Y_M(t-\ell z_j,x-v\ell z_j).$$
        $Y_b$ is the known initial or incoming value. This imposes the initial
        and incoming distributions directly; no guessed boundary penalty is used.

        The source must agree with moments of its reconstructed distribution:
        $$r_\rho=\rho_\theta-\int g\,dv,\qquad
        r_p=\rho_\theta u_\theta-\int vg\,dv,$$
        $$r_E=\frac12\rho_\theta(u_\theta^2+3T_\theta)-
        \frac12\int(v^2g+h)\,dv.$$
        At $N$ fixed space-time collocation points, minimize
        $$\mathcal L=\sum_{j\in\{\rho,p,E\}}\lambda_j
        \frac1N\sum_{i=1}^{N}(r_j(t_i,x_i)/s_j)^2.$$
        $\lambda_j$ are student-chosen weights; $s_\rho=\rho_L=1$,
        $s_p=\rho_L\sqrt{T_L}=1$, $s_E=\rho_LT_L=1$ are fixed upstream scales.
        Changing a weight changes the tradeoff, not the unit or definition of a
        residual. A zero weight is an **ablation that omits part of BGK consistency**.

        Autograd differentiates these quadrature sums with respect to network
        parameters. This is an integral characteristic PINN, not a supervised fit
        to DVM values. See [the small implementation](../../flowmllab/kinetic_pinn.py)
        for the reconstruction, loss and optimizer loops.
    ''')
    code('w10-2-fresh-controls', '''
        # EDIT HERE. Start each experiment by rerunning this cell and the training cell.
        # A hidden depth is the number of tanh hidden layers, not the output layer.
        DEPTH = 2                       # try 1, 2, 3; allowed 1..5
        WIDTH = 24                      # neurons per hidden layer
        OPTIMIZER = 'adam_lbfgs'         # 'adam', 'lbfgs', or 'adam_lbfgs'
        SEED = 1729
        LOSS_WEIGHTS = {'density':1.0, 'momentum':1.0, 'energy':1.0}
        ADAM_STEPS, LBFGS_STEPS = 100, 20
        LEARNING_RATE = 0.003
        COLLOCATION_POINTS = 64
        VELOCITY_ORDER, TIME_ORDER = 32, 8
        MAX_SECONDS = 120.0              # safety ceiling; inspect the actual stop reason
        config = kp.PINNConfig(seed=SEED,depth=DEPTH,width=WIDTH,optimizer=OPTIMIZER,
            weights=tuple(LOSS_WEIGHTS[k] for k in kp.COMPONENTS),
            adam_steps=ADAM_STEPS,lbfgs_steps=LBFGS_STEPS,learning_rate=LEARNING_RATE,
            collocation_points=COLLOCATION_POINTS,velocity_order=VELOCITY_ORDER,
            time_order=TIME_ORDER,vmax=7.0,max_seconds=MAX_SECONDS,threads=1)
        display(pd.DataFrame([asdict(config)]))
    ''')
    md('w10-2-fresh-fit-notes', r'''
        ### Optimizer work and a fair comparison

        Adam uses full-batch gradients here; L-BFGS uses a strong-Wolfe line
        search on the same fixed collocation set. A line search may evaluate the
        loss/gradient several times in one outer step. Compare **actual gradient
        evaluations and elapsed time**, not just outer iteration counts. Report
        setup time separately because first-use library imports can be expensive.

        `kp.train(config)` always creates a fresh model. It returns the **last
        completed update**, never the best DVM score. Network initialization and
        collocation have separate seeded random generators, so changing depth
        does not silently change the physics points. A separate fixed set of
        space-time points is evaluated before and after training as a physics
        audit; it is not used for updates or checkpoint selection. DVM field
        errors are computed later. Once you use them to tune a model, label that
        reference as development information, not an unseen test.
    ''')
    code('w10-2-fresh-train', '''
        trained = kp.train(config)  # fresh random initialization, CPU; no DVM argument
        training_report = kp.training_report(trained)
        display(pd.DataFrame([trained.work]))
        print('Stop:',trained.stop_reason)
        print('Initial/final training loss:',trained.history[0]['loss'],trained.history[-1]['loss'])
        print('Frozen physics audit (not used for fitting):')
        display(pd.DataFrame({k:v for k,v in trained.training_diagnostics.items()
                             if isinstance(v,dict)}).T)
        print('Initial state hash:',trained.provenance['initial_state_sha256'])
        print('Collocation hash:',trained.provenance['collocation_sha256'])
    ''')
    code('w10-2-fresh-loss-figure', '''
        fig,axes=plt.subplots(1,2,figsize=(10,3.6))
        for component,color in zip(kp.COMPONENTS,('#2355b5','#c52a32','#16834b')):
            values=[h['scaled_mse'][component] for h in trained.history]
            axes[0].semilogy([h['gradient_evaluations'] for h in trained.history],values,
                             color=color,label=component)
            axes[1].semilogy([h['elapsed_seconds'] for h in trained.history],values,color=color,label=component)
        axes[0].set_xlabel('Gradient evaluations'); axes[1].set_xlabel('Elapsed training time (s)')
        for ax in axes:
            ax.set_ylabel('Unweighted scaled consistency MSE'); ax.grid(alpha=.18)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
        fig.tight_layout(rect=(0,0,1,.86)); plt.show()
        print('Plot unweighted components even when weights change: a smaller weighted sum alone is not a fair accuracy claim.')
    ''')
    md('w10-2-fresh-dvm-notes', r'''
        ## 11. Independent DVM and Table H: initial versus trained PINN

        The reference solves $\partial_tY+v\partial_xY=(Y_M-Y)/\tau$ for all
        three reduced distributions with a finite-volume minmod reconstruction,
        upwind face fluxes and two-stage SSPRK2. Incoming half-range face values
        use the exact reservoirs; outgoing values come from the interior.
        Each stage recomputes its own Maxwellian from its moments.
        The time-step bound is
        $\Delta t\leq C/(\max|v|/\Delta x+1/\tau)$ with $C=0.4$.

        This is an independently implemented NumPy solver, not the trained
        network's output. Both methods use analytic transverse Maxwellian
        integration and longitudinal Gauss-Legendre quadrature on $[-7,7]$.
        Training uses order 32; assessment uses 48. No clipping, moment correction,
        fitted spatial shift or amplitude rescaling is applied.

        Run 400 and 800 spatial cells at the same CFL. Their comparison checks
        **joint space/time sensitivity**, not a measured convergence order or GCI.
        Positivity, the necessary condition $gk-h^2\geq0$, and separate boundary
        and collision balance diagnostics are reported. Small algebraic balance
        does not by itself certify physical accuracy. The paper's original finite
        transverse-velocity quadrature is a different discretization; its saved
        curves above retain their own identity.
    ''')
    code('w10-2-fresh-dvm', '''
        # These runs occur AFTER the model has been trained and frozen.
        dvm400=kd.solve(kd.DVMConfig(cells=400,velocity_order=48,cfl=0.4))
        dvm800=kd.solve(kd.DVMConfig(cells=800,velocity_order=48,cfl=0.4))
        display(pd.DataFrame([{'cells':n,**r['diagnostics']}
                              for n,r in [(400,dvm400),(800,dvm800)]]))
        x_eval=dvm800['x']
        evaluation_rule=kp.Quadrature(velocity_order=48,time_order=12,vmax=7.)
        initial=kp.evaluate(trained,x_eval,quadrature=evaluation_rule,state='initial')
        prediction=kp.evaluate(trained,x_eval,quadrature=evaluation_rule)
        def field_errors(x,pred,ref):
            return {f:ks.profile_metrics(x,pred[:,i],ref[:,i])['relative_l2_percent']
                    for i,f in enumerate(kp.FIELDS)}
        table_H=pd.DataFrame({'initial_L2_percent':field_errors(x_eval,initial['fields'],dvm800['fields']),
                              'trained_L2_percent':field_errors(x_eval,prediction['fields'],dvm800['fields'])})
        table_H['trained_minus_initial']=table_H.trained_L2_percent-table_H.initial_L2_percent
        display(table_H.rename_axis('Table H: field'))
        print('Source/distribution consistency RMS (rho, momentum, physical total energy):',prediction['consistency_rms'])
        print('All plotted fields are moments of reconstructed g/h/k, not source-network outputs.')
    ''')
    code('w10-2-fresh-physical-figure', '''
        fresh_labels={'rho':r'$\\rho$','u':r'$u$','T':r'$T$','q':r'$q_x$',
                      'sigma':r'$\\sigma_{xx}$','Delta':r'$\\Delta$'}
        fig,axes=plt.subplots(2,3,figsize=(10,6))
        for ax,(field,label) in zip(axes.flat,fresh_labels.items()):
            i=kp.FIELDS.index(field)
            ax.plot(x_eval,dvm800['fields'][:,i],'k-',lw=2,label='Independent BGK DVM')
            ax.plot(x_eval,initial['fields'][:,i],color='#2355b5',ls='--',label='Random-start PINN')
            ax.plot(x_eval,prediction['fields'][:,i],color='#c52a32',ls='-.',label='Trained CPU PINN')
            ax.set(xlabel='x',ylabel=label,xlim=(-.5,.5)); ax.grid(alpha=.16)
            ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,3),useMathText=True)
        fig.legend(*axes.flat[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.87)); plt.show()
        fig,axes=plt.subplots(1,3,figsize=(10,3.6))
        for ax,field in zip(axes,('q','sigma','Delta')):
            i=kp.FIELDS.index(field); scale=np.max(np.abs(dvm800['fields'][:,i]))
            for values,color,style,label in [(initial,'#2355b5','--','Initial minus DVM'),
                                              (prediction,'#c52a32','-','Trained minus DVM')]:
                ax.plot(x_eval,(values['fields'][:,i]-dvm800['fields'][:,i])/scale,
                        color=color,ls=style,label=label)
            ax.axhline(0,color='0.4',lw=.6); ax.set(xlabel='x',ylabel=fresh_labels[field]+' signed discrepancy')
            ax.grid(alpha=.16)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.84)); plt.show()
        print("Each signed discrepancy uses that field's full-reference maximum magnitude, never a local near-zero denominator.")
    ''')
    md('w10-2-fresh-sensitivity-notes', r'''
        ## 12. Table I: resolve numerical sensitivity before claiming improvement

        Restrict the 800-cell **distributions** conservatively to 400 cells before
        forming moments. Averaging already-central moments would be a different
        operation. At fixed 800 cells, change velocity order from 48 to 64.
        Separately reevaluate the frozen PINN with orders $(N_v,N_t)=(64,16)$;
        no retraining occurs. Compare these variations with the reported PINN
        errors. They are diagnostics, not uncertainty bounds from independent
        stochastic realizations. A small two-level change does not prove convergence.
    ''')
    code('w10-2-fresh-sensitivity', '''
        reduced=dvm800['state'].reshape(400,2,3,len(dvm800['v'])).mean(axis=1)
        restricted=kd.moments(reduced,dvm800['v'],dvm800['weights'])
        restricted_fields=np.column_stack([restricted[f] for f in kp.FIELDS])
        dvm_v64=kd.solve(kd.DVMConfig(cells=800,velocity_order=64,cfl=0.4))
        refined_prediction=kp.evaluate(trained,x_eval,quadrature=kp.Quadrature(64,16,7.))
        table_I=pd.DataFrame({
            'DVM_space_time_400_to_800_percent':field_errors(dvm400['x'],dvm400['fields'],restricted_fields),
            'DVM_velocity_48_to_64_percent':field_errors(x_eval,dvm800['fields'],dvm_v64['fields']),
            'PINN_quadrature_48x12_to_64x16_percent':field_errors(x_eval,prediction['fields'],refined_prediction['fields'])})
        display(table_I.rename_axis('Table I: field'))
    ''')
    md('w10-2-fresh-study-notes', r'''
        ## 13. Table J: a controlled optimizer, depth and weight study

        The next cell can run seven **fresh** configurations. It is off by default
        to keep Run All bounded. The published seven-run example, including all
        initial/final curves, seeds, losses and work counts, is linked below.
        Neither its reference scores nor this optional cell choose a winner to
        use as the default. All configurations were declared before DVM scoring.

        Start from a clean baseline: same seed, points, width, learning rate and
        quadrature. Adam gets 120 updates; L-BFGS gets 40 outer steps; the hybrid
        gets 100+20. These **are not equal-work budgets**. Record gradient and total
        function evaluations and elapsed time. Depth changes parameter count and
        initialization shape; it is a capacity experiment, not just an optimizer test.
        The energy weight changes to 0.1 or 10 while density/momentum stay at 1.

        [Published protocol](../../results/week10_2_kinetic_shock/pinn_training_protocol.json)
        · [Nine-field scores](../../results/week10_2_kinetic_shock/pinn_cpu_metrics.csv)
        · [Full work/loss report](../../results/week10_2_kinetic_shock/pinn_cpu_report.json)
        · [Physical optimizer comparison](../../results/week10_2_kinetic_shock/pinn_optimizer_profiles.png).
    ''')
    code('w10-2-fresh-study', '''
        RUN_CONTROLLED_STUDY = False
        baseline=kp.PINNConfig(seed=1729,depth=2,width=24,optimizer='adam_lbfgs',
            adam_steps=100,lbfgs_steps=20,max_seconds=120.,threads=1)
        study_configs={'default':baseline,
            'adam':replace(baseline,optimizer='adam',adam_steps=120,lbfgs_steps=0),
            'lbfgs':replace(baseline,optimizer='lbfgs',adam_steps=0,lbfgs_steps=40),
            'depth1':replace(baseline,depth=1),'depth3':replace(baseline,depth=3),
            'energy0p1':replace(baseline,weights=(1.,1.,.1)),
            'energy10':replace(baseline,weights=(1.,1.,10.))}
        if RUN_CONTROLLED_STUDY:
            # Finish all fits before scoring any of them against the DVM.
            study_runs={name:kp.train(cfg) for name,cfg in study_configs.items()}
            rows=[]
            for name,run in study_runs.items():
                values=kp.evaluate(run,x_eval,quadrature=evaluation_rule)
                errors=field_errors(x_eval,values['fields'],dvm800['fields'])
                rows.append({'run':name,'stop':run.stop_reason,**run.work,
                    'final_physics_loss':run.history[-1]['loss'],
                    **{f+'_L2_percent':e for f,e in errors.items()}})
            table_J=pd.DataFrame(rows).set_index('run')
            display(table_J)
            fig,axes=plt.subplots(1,2,figsize=(10,4))
            for ax,field in zip(axes,('q','sigma')):
                i=kp.FIELDS.index(field)
                ax.plot(x_eval,dvm800['fields'][:,i],'k-',lw=2,label='DVM')
                for name in ('adam','lbfgs','default'):
                    values=kp.evaluate(study_runs[name],x_eval,quadrature=evaluation_rule)
                    ax.plot(x_eval,values['fields'][:,i],ls='--',label=name)
                ax.set(xlabel='x',ylabel=fresh_labels[field],xlim=(-.25,.25)); ax.grid(alpha=.16)
            fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
            fig.tight_layout(rect=(0,0,1,.80)); plt.show()
        else:
            print('Table J sweep is optional. Set RUN_CONTROLLED_STUDY=True to train all seven cases.')
            print('The default run above was trained fresh; published sweep results have a separate receipt.')
        print('Total notebook elapsed time, including imports, figures and fresh solvers:',
              round(time.perf_counter()-START,2),'s')
    ''')
    md('w10-2-fresh-exercises', r'''
        ## 14. Submit the from-scratch PINN experiments

        Continue the seven investigations above with these five concrete tasks.
        Your submission must identify which outputs you freshly generated and
        which published results you read. Keep the physical problem fixed.

        8. **Depth (Tables H/J).** Train depths 1, 2 and 3 at width 24 with seed
           1729 and the same physics points. Report parameters, completed updates,
           work/time, training/audit residuals and all nine DVM errors. Overlay
           signed heat flux and stress with DVM. Repeat with seed 1730 and explain
           whether a claimed depth advantage survives. Do not equate capacity
           with automatic accuracy.
        9. **Optimizer (Tables H/J).** Compare Adam, L-BFGS and Adam then L-BFGS.
           Verify identical initial-state and collocation hashes. Report actual
           gradient/function counts, setup/update time and stop reasons; plot
           component losses against work and time. Rerun with closer work budgets
           if making an efficiency claim. Explain why an outer L-BFGS step is not
           comparable to one Adam step and why a switching kink need not be a bug.
        10. **Weights (Tables H/J).** Set energy weight to 0.1, 1 and 10. Report
            **unweighted** density/momentum/energy residuals as well as the
            weighted sum. Plot $q,\sigma,\Delta$ and their signed errors. Identify
            which fields improve and deteriorate. Optional energy weight 0 is an
            incomplete-consistency ablation, not the original BGK model.
        11. **Sampling and quadrature (Tables H/I).** Compare 32, 64 and 128
            collocation points at fixed depth/seed, then one change of velocity
            or characteristic quadrature. Separate fitting error from integration
            error by reevaluating the same frozen model. Report DVM space/time
            and velocity sensitivity before attributing a small gain to training.
        12. **Did the network learn? (Tables H/I).** Quantify initial-to-trained
            changes in all nine fields and in independent physics-audit residuals.
            Explain the small collision fraction at $t_f/\tau=0.02$. Use the
            actual signed profiles to defend one improvement and one limitation;
            a decrease in weighted training loss alone is insufficient.

        **Deliverables:** the filled tables, configuration dictionaries, seed and
        hashes, work/time and stop reasons, readable physical overlays, and a
        concise numerical interpretation. Retain failed cases. DVM data must not
        enter the loss or select a checkpoint. If you use DVM scores to tune
        hyperparameters, disclose that development role and reserve new conditions
        before making a generalization claim. This short CPU lab is not evidence
        for long-time, stationary or Shakhov performance.
    ''')
