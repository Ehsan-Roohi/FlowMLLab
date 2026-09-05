#!/usr/bin/env python3
"""CPU/Kokkos step comparison. Execution evidence, never a training-data release."""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import tempfile
import time

spec = importlib.util.spec_from_file_location('step_campaign', Path(__file__).with_name('campaign.py'))
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
pilot = campaign.pilot
POINTER = 'LATEST_SPARTA_STEP_GPU_BENCHMARK'
FILES = ['gpu_benchmark.py', 'gpu_job.sh', 'submit_gpu_benchmark.sh', 'cuda_probe.cu',
         'verify_gpu.py', 'GPU.md', 'campaign.py', 'pilot.py']
ARCH = {'a40': ('8.6', 'AMPERE86'), 'a100': ('8.0', 'AMPERE80'), 'h100': ('9.0', 'HOPPER90')}


def load(path):
    return json.loads(Path(path).read_text())


def kk_deck(path):
    """Require accelerated implementations for particle/collision computes.

    ave/time and ave/surf remain host fixes at this source pin. Their synchronization
    costs belong in the measured time, not in an assumed speedup.
    """
    # At this pin the wall dispatch compares sc->style with "diffuse", so writing
    # diffuse/kk explicitly fails. -sf kk selects SurfCollideDiffuseKokkos while
    # retaining the canonical style name. Its kokkosable flag is checked upstream.
    mappings = {'collide': (1, {'vss'}),
                'compute': (2, {'boundary', 'grid', 'thermal/grid', 'surf'}),
                'fix': (2, {'emit/face', 'grid/check', 'ave/grid'})}
    result = []
    for line in Path(path).read_text().splitlines():
        fields = line.split()
        if fields and fields[0] in mappings:
            index, styles = mappings[fields[0]]
            if len(fields) > index and fields[index] in styles:
                fields[index] += '/kk'
                line = ' '.join(fields)
        result.append(line)
    Path(path).write_text('\n'.join(result) + '\n')


def command(binary, backend, launcher='mpirun', ranks=16, host_kokkos=False, serial=False):
    result = [] if serial else [launcher, '-np', str(1 if backend == 'kokkos' else ranks), '--bind-to', 'core']
    result += [str(binary)]
    if backend == 'kokkos':
        result += ['-k', 'on']
        if not host_kokkos:
            result += ['g', '1']
        result += ['-sf', 'kk']
        if not host_kokkos:
            result += ['-pk', 'kokkos', 'gpu/aware', 'off']
    return result


def execute(path, cmd, timeout=7200, gpu_uuid=None):
    path = Path(path)
    monitor = None
    with open(path/'in.step') as inp, open(path/'solver.stdout', 'w') as output, \
         open(path/'gpu-memory.csv', 'w') as memory:
        if gpu_uuid:
            monitor = subprocess.Popen(['nvidia-smi', '-i', gpu_uuid,
                '--query-gpu=timestamp,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits', '--loop-ms=500'],
                stdout=memory, stderr=subprocess.DEVNULL)
        start = time.monotonic()
        try:
            completed = subprocess.run(cmd, cwd=path, stdin=inp, stdout=output,
                                       stderr=subprocess.STDOUT, timeout=timeout)
        finally:
            if monitor:
                monitor.terminate()
                monitor.wait(timeout=10)
        elapsed = time.monotonic()-start
    log = (path/'solver.stdout').read_text(errors='replace')
    if completed.returncode:
        print(log[-10000:], file=sys.stderr)
        raise RuntimeError(f'Solver failed rc={completed.returncode}: {path}')
    stuck = [int(v) for v in re.findall(r'Particles stuck\s*=\s*(\d+)', log)]
    if re.search(r'^ERROR', log, re.M) or not stuck or any(stuck):
        raise ValueError(f'Solver error or stuck-particle accounting invalid: {path}')
    loops = re.findall(r'Loop time of ([\d.eE+-]+) on (\d+) procs for (\d+) steps', log)
    if not loops:
        raise ValueError('Solver timing lines absent')
    memory_samples = []
    for line in (path/'gpu-memory.csv').read_text().splitlines():
        values = line.split(',')
        if len(values) == 3:
            with contextlib.suppress(ValueError):
                memory_samples.append(float(values[1]))
    timing = dict(wall_seconds=elapsed,
                  loops=[dict(seconds=float(t), ranks=int(n), steps=int(s)) for t, n, s in loops],
                  peak_device_memory_sampled_mib=max(memory_samples, default=None),
                  memory_note='500 ms samples of allocated device usage, not an exact allocation peak.',
                  command=cmd)
    pilot.write_json(path/'timing.json', timing)
    return timing


def smoke_row(height=50):
    row = campaign.matrix()[6].copy()
    row.update(id=f'gpu_smoke_h{height}', height_percent=height, nx=20, ny=100, ppc=8,
               warmup_steps=100, sampling_steps=400, block_steps=50, sample_every=10)
    return row


def validate_case(path):
    with contextlib.redirect_stdout(io.StringIO()):
        result = campaign.summarize_case(path)
    if result['stuck_particles'] != 0:
        raise ValueError('Stuck particles')
    return result


def preflight(out, cpu_binary, kk_binary, launcher='mpirun', host_kokkos=False, serial=False):
    root = Path(out)/'preflight'
    root.mkdir()
    launch = {b: command(cpu_binary if b == 'cpu' else kk_binary, b, launcher,
                        ranks=2, host_kokkos=host_kokkos, serial=serial) for b in ['cpu', 'kokkos']}
    for height in [16, 50, 75]:
        for backend in ['cpu', 'kokkos']:
            path = root/f'{backend}-fresh-h{height}'
            campaign.generate_case(path, smoke_row(height), smoke=True)
            if backend == 'kokkos': kk_deck(path/'in.step')
            execute(path, launch[backend], timeout=300)
            validate_case(path)
            print(f'GPU_ROUTE_FRESH_PASS backend={backend} height={height}', flush=True)
    # Both directions matter: existing CPU particles -> GPU, and GPU checkpoint -> CPU.
    for previous, backend in [('cpu', 'kokkos'), ('kokkos', 'cpu')]:
        path = root/f'{previous}-to-{backend}'
        campaign.generate_case(path, smoke_row(), root/f'{previous}-fresh-h50'/'restart.final',
                               'smoke', smoke=True)
        if backend == 'kokkos': kk_deck(path/'in.step')
        execute(path, launch[backend], timeout=300)
        validate_case(path)
        print(f'GPU_ROUTE_RESTART_PASS from={previous} to={backend}', flush=True)
    pilot.write_json(Path(out)/'preflight.json', dict(status='preflight_complete',
        kokkos_backend='Serial' if host_kokkos else 'CUDA',
        actual_cuda_test=not host_kokkos, training_data_approved=False))


def benchmark_row(seed):
    row = campaign.matrix()[0].copy()
    row.update(id=f'gpu_benchmark_s{seed}', seed=seed, warmup_steps=700,
               sampling_steps=1400, block_steps=350, sample_every=35)
    return row


def capacity_deck(path, row):
    """Allocate campaign-scale particles and all production tallies, then move 70 steps.

    Fresh, transient flow: this proves a memory/load path, not steady-state capacity
    or scientific convergence. No enormous field/restart dump is produced.
    """
    path = Path(path)
    m = pilot.generate(path, smoke=True, ratio=row['height_percent']/100, nx=row['nx'],
                       ny=row['ny'], ppc=row['ppc'], seed=row['seed'])
    prefix = (path/'in.step').read_text().split('fix check grid/check ', 1)[0]
    deck = campaign.sampling_deck(dict(warmup_steps=35, sampling_steps=70,
                                      block_steps=35000, sample_every=35))
    # Use the production tally cadence; run lengths need not emit a completed block.
    deck = deck.split('print "SPARTA_STEP_SAMPLING_BEGIN"', 1)[1]
    lines = []
    for line in deck.splitlines():
        if line.startswith(('run ', 'dump ', 'dump_modify ', 'write_restart ', 'print ')):
            continue
        lines.append(line)
    path.joinpath('in.step').write_text(prefix+f'timestep {row["dt_s"]:.17g}\n'
        'compute boundary boundary gas n press shx ke\n'
        'variable exits_in equal c_boundary[1][1]\n'
        'variable exits_out equal c_boundary[2][1]\n'
        'variable inventory equal np\n'
        'fix check grid/check 35 error\n'
        'stats 35\nstats_style step cpu np ncoll nattempt nexit\n'
        +'\n'.join(lines)+'\nrun 70\nprint "GPU_CAPACITY_PROBE_COMPLETE"\n')
    kk_deck(path/'in.step')
    m.update(level='transient_gpu_capacity_probe', dt_s=row['dt_s'], final_step=70,
             warmup_steps=0, sampling_steps=70, campaign_case=row,
             training_data_approved=False)
    pilot.write_json(path/'case.json', m)


def run(out):
    out = Path(out).resolve()
    cfg = load(out/'manifest.json')
    hardware = load(out/'cuda-device.json')
    cpu, gpu = out/'build-cpu'/'src'/'spa_mpi', out/'build-gpu'/'src'/'spa_kokkos_cuda'
    launcher = cfg['mpi_launcher']
    preflight(out, cpu, gpu, launcher)
    restart = Path(cfg['restart'])
    # Hash once, then require exactly the same restart for every timed arm.
    restart_hash = pilot.sha(restart)
    summary = dict(status='benchmark_running', flowmllab_commit=cfg['flowmllab_commit'],
        sparta_commit=pilot.SPARTA_COMMIT, hardware=hardware, restart=str(restart),
        restart_sha256=restart_hash, measurements=[], comparisons=[], capacity=[],
        training_data_approved=False, scientific_cpu_gpu_equivalence_validated=False,
        notes=['Three paired repeat runs begin at the same CPU pilot restart; they are not independent flow datasets.',
               'Seed labels do not imply identical CPU/GPU random trajectories.',
               'Cell sampling cadence and dt match campaign; benchmark block dumps occur 100x more often.',
               'Warmup loop timing measures pressure-driven transport/collisions before field tallies.',
               'Sampling loop timing includes all tallies and deliberately frequent compressed output.',
               'End-to-end timing includes input, MPI startup, restart read/write and output.',
               'Short-window field differences are diagnostics, not a statistical equivalence test.',
               'Fine-grid capacity probes start fresh. Long-run memory headroom still needs observation.'])
    pilot.write_json(out/'gpu_benchmark_report.json', summary)
    for i, seed in enumerate(campaign.SEEDS):
        reports, timings, paths = {}, {}, {}
        # Alternate arm order to reduce warm-cache/order bias.
        for backend in (['cpu', 'kokkos'] if i % 2 == 0 else ['kokkos', 'cpu']):
            path = out/f'benchmark-{backend}-s{seed}'
            row = benchmark_row(seed)
            campaign.generate_case(path, row, restart, 'smoke', smoke=True)
            if backend == 'kokkos': kk_deck(path/'in.step')
            m = load(path/'case.json')
            m.update(level='cpu_gpu_short_benchmark', training_data_approved=False)
            pilot.write_json(path/'case.json', m)
            cmd = command(cpu if backend == 'cpu' else gpu, backend, launcher)
            timing = execute(path, cmd, gpu_uuid=hardware['uuid'] if backend == 'kokkos' else None)
            reports[backend] = validate_case(path)
            if [x['steps'] for x in timing['loops']] != [700, 1050, 350]:
                raise ValueError('Unexpected benchmark loop segmentation')
            timing.update(backend=backend, seed=seed,
                          warmup_loop_seconds=timing['loops'][0]['seconds'],
                          sampling_loop_seconds=sum(v['seconds'] for v in timing['loops'][1:]))
            summary['measurements'].append(timing)
            timings[backend], paths[backend] = timing, path
            pilot.write_json(out/'gpu_benchmark_report.json', summary)
            print(f'BENCHMARK_ARM_COMPLETE backend={backend} seed={seed} wall_s={timing["wall_seconds"]:.3f}', flush=True)
        # diagnostic_probes reads the verified raw cumulative dump and uses fixed physical bins.
        probes = {}
        for b, p in paths.items():
            probes[b] = {(r[0],r[1]):dict(zip(['x_m','y_m','area_m2','u_m_s','v_m_s','p_Pa'],r[2:]))
                         for r in campaign.diagnostic_probes(p/'grid.final.gz', load(p/'case.json'))}
        # field_difference uses its second argument as the reference norm.
        difference = campaign.field_difference(probes['kokkos'], probes['cpu'])
        summary['comparisons'].append(dict(seed=seed, field_difference=difference,
            mass_flow_cpu=reports['cpu']['mass_out_kg_per_m_s'],
            mass_flow_gpu=reports['kokkos']['mass_out_kg_per_m_s'],
            boundary_pressure_cpu=reports['cpu']['boundary_adjacent_mean_pressure_Pa'],
            boundary_pressure_gpu=reports['kokkos']['boundary_adjacent_mean_pressure_Pa']))
        pilot.write_json(out/'gpu_benchmark_report.json', summary)
    if pilot.sha(restart) != restart_hash:
        raise ValueError('Pilot restart changed during paired benchmark')
    summary['speedups'] = {}
    for key in ['warmup_loop_seconds', 'sampling_loop_seconds', 'wall_seconds']:
        med = {b: statistics.median(r[key] for r in summary['measurements'] if r['backend']==b)
               for b in ['cpu', 'kokkos']}
        summary['speedups'][key] = dict(cpu_16_ranks_median_s=med['cpu'], gpu_1_median_s=med['kokkos'],
                                       cpu_over_gpu=med['cpu']/med['kokkos'])
    summary['status'] = 'paired_benchmark_complete_capacity_pending'
    pilot.write_json(out/'gpu_benchmark_report.json', summary)
    # The densest production case is fine h50/PPC40, and h16 maximizes the PPC20 geometry sweep.
    for row in [dict(campaign.matrix()[11]), dict(campaign.matrix()[10])]:
        path = out/f'capacity-{row["id"]}'
        capacity_deck(path, row)
        try:
            timing = execute(path, command(gpu, 'kokkos', launcher), timeout=3600, gpu_uuid=hardware['uuid'])
            if 'GPU_CAPACITY_PROBE_COMPLETE' not in (path/'solver.stdout').read_text():
                raise ValueError('Capacity marker missing')
            entry = dict(case_id=row['id'], status='transient_capacity_pass', timing=timing,
                         initial_particles_estimate=row['initial_simulated_particles_estimate'])
        except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            entry = dict(case_id=row['id'], status='capacity_probe_failed', reason=str(exc))
            summary['capacity'].append(entry)
            summary['status'] = 'paired_benchmark_complete_capacity_failed'
            pilot.write_json(out/'gpu_benchmark_report.json', summary)
            raise
        summary['capacity'].append(entry)
        pilot.write_json(out/'gpu_benchmark_report.json', summary)
        print(f'GPU_CAPACITY_PASS case={row["id"]}', flush=True)
    summary['status'] = 'gpu_route_benchmark_complete'
    pilot.write_json(out/'gpu_benchmark_report.json', summary)
    (out/'GPU_BENCHMARK_COMPLETE').write_text('Execution/performance evidence only; campaign not submitted.\n')
    show_report(summary)
    print('SPARTA_GPU_BENCHMARK_COMPLETE TRAINING_DATA_APPROVED=False', flush=True)


def show_report(report):
    print('STATUS=', report['status'])
    for name, values in report.get('speedups', {}).items():
        print(f'{name}: CPU16={values["cpu_16_ranks_median_s"]:.3f}s GPU1={values["gpu_1_median_s"]:.3f}s CPU/GPU={values["cpu_over_gpu"]:.3f}')
    for entry in report.get('capacity', []):
        print('CAPACITY=', entry['case_id'], entry['status'], entry.get('timing', {}).get('peak_device_memory_sampled_mib'))


def submit(root, base, ref, gpu='a40', new_run=False):
    if not re.fullmatch('[0-9a-f]{40}', ref): raise ValueError('Full commit SHA required')
    root, base = Path(root).resolve(), Path(base).resolve()
    (base/'runs').mkdir(parents=True, exist_ok=True)
    if (base/POINTER).exists() and not new_run:
        raise ValueError(f'Benchmark already recorded: {base/POINTER}. Use status or --new-run intentionally.')
    restart = base/'runs'/campaign.PILOT_RUN/'pilot'/'restart.final'
    if not restart.is_file() or not restart.stat().st_size:
        raise ValueError(f'Paired benchmark needs the successful CPU pilot restart: {restart}')
    previous = load(restart.parent/'case.json')
    if (previous['sparta_commit'] != pilot.SPARTA_COMMIT or previous['final_step'] != 60000
        or previous['nx'] != 1000 or previous['ny'] != 200 or previous['h_over_H'] != .5
        or previous['ppc_outlet_reference'] != 20):
        raise ValueError('Unexpected pilot restart metadata')
    for name in FILES:
        if not (root/name).is_file(): raise ValueError(f'Missing source file {name}')
    if shutil.disk_usage(base).free < 30*2**30:
        raise ValueError('Need 30 GiB free filesystem headroom for build and comparison outputs')
    out = Path(tempfile.mkdtemp(prefix=time.strftime('step-gpu-benchmark-%Y%m%dT%H%M%SZ-',time.gmtime()), dir=base/'runs'))
    code = out/'code'; code.mkdir()
    for name in FILES: shutil.copy2(root/name, code/name)
    (out/'code.sha256').write_text(''.join(f'{pilot.sha(code/n)}  code/{n}\n' for n in FILES))
    meta = dict(status='submitting', flowmllab_commit=ref, sparta_commit=pilot.SPARTA_COMMIT,
        restart=str(restart), gpu_type=gpu, expected_compute_capability=ARCH[gpu][0],
        kokkos_arch=ARCH[gpu][1], cuda_module='cuda/12.6', mpi_module='openmpi/5.0.3-cuda12.6',
        out=str(out), cpu_ranks=16, gpu_ranks=1, jobs={}, training_data_approved=False)
    pilot.write_json(out/'manifest.json', meta)
    (base/POINTER).write_text(str(out)+'\n')
    cmd = ['sbatch', '--parsable', '--account=pi_roohie_umass_edu', '--partition=gpu',
        '--nodes=1', '--ntasks=16', '--cpus-per-task=1', '--gpus=1', f'--constraint={gpu}',
        '--mem=128G', '--time=04:00:00', '--job-name=step-gpu-bench', '--export=ALL',
        f'--chdir={out}', f'--output={out}/slurm-%j.out', f'--error={out}/slurm-%j.err', str(code/'gpu_job.sh')]
    try:
        raw = subprocess.check_output(cmd, env=dict(os.environ, SPARTA_GPU_OUT=str(out)), text=True).strip()
        job = raw.split(';')[0]
        if not job.isdigit(): raise ValueError(f'Invalid sbatch result: {raw!r}')
        meta.update(status='submitted', jobs={'benchmark':job})
        (out/'JOB_ID').write_text(job+'\n')
    except Exception:
        meta['status']='submission_failed'; pilot.write_json(out/'manifest.json', meta); raise
    pilot.write_json(out/'manifest.json', meta)
    print(f'OUT={out}\nGPU_BENCHMARK_JOB={job}\nGPU={gpu} CPU_REFERENCE_RANKS=16')
    print('Only the benchmark is submitted. Existing campaigns are unchanged.')


def status(out):
    out = Path(out)
    cfg = load(out/'manifest.json')
    print('OUT=', out)
    job = cfg['jobs'].get('benchmark')
    if job:
        print('JOB=', job)
        subprocess.run(['sacct', '-j', job, '--format=JobID%20,State%22,Elapsed,ExitCode,MaxRSS,NodeList%20'], check=False)
    path = out/'gpu_benchmark_report.json'
    if path.exists(): show_report(load(path))
    else: print('BENCHMARK_REPORT_NOT_YET_AVAILABLE')
    for path in sorted(out.glob('slurm-*.out'))+sorted(out.glob('slurm-*.err')):
        print('LOG=',path)
        subprocess.run(['tail','-n','25',str(path)], check=False)


def pack(out):
    out = Path(out).resolve()
    target = out/'gpu_benchmark_review.tar.gz'
    with tarfile.open(target, 'w:gz') as archive:
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path == target: continue
            relative=path.relative_to(out)
            if relative.parts[0] in ['source', 'build-cpu', 'build-gpu', 'cmake-tools']: continue
            if path.name.startswith(('restart.', 'grid.', 'wall.')) or path.name=='fields.csv.gz': continue
            if path.stat().st_size < 5*2**20:
                archive.add(path, arcname=str(relative))
    print('REVIEW_PACK=', target)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    s=p.add_subparsers(dest='action', required=True)
    q=s.add_parser('submit'); q.add_argument('--root', type=Path, default=Path(__file__).parent)
    q.add_argument('--base', default=pilot.DEFAULT_BASE); q.add_argument('--ref', required=True)
    q.add_argument('--gpu', choices=list(ARCH), default='a40'); q.add_argument('--new-run', action='store_true')
    for name in ['run','status','pack']:
        s.add_parser(name).add_argument('--out', type=Path, required=True)
    a=p.parse_args()
    if a.action=='submit': submit(a.root,a.base,a.ref,a.gpu,a.new_run)
    elif a.action=='run': run(a.out)
    elif a.action=='status': status(a.out)
    elif a.action=='pack': pack(a.out)


if __name__=='__main__': main()
