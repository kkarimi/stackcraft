# Independent final evidence audit

This archives the exact CPU audit script run after evaluation. Its SHA256 is
recorded in [the audit result](final-audit.json). It assumes the original local
`runs/`, dataset and checkpoint layout; it is an execution record, not a portable
release-verification CLI. Use `scripts/verify_release.py` for release downloads.
No new games or model inference were run by this audit.

```text
"""Independent CPU-only audit of frozen final evidence; reconstructs saved actions only."""
import hashlib
import importlib.util
import json
import math
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path.cwd()
OUT = ROOT / 'runs/final-review/audit.json'
assert not OUT.exists(), 'Refuse to overwrite audit'
start = time.monotonic()
def read(p):
    return json.loads(Path(p).read_text())
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while chunk := f.read(1048576):
            h.update(chunk)
    return h.hexdigest()
def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / name)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

evaluator = module('evaluate_clef.py')
exporter = module('export_demo.py')
request = read('runs/final-evaluation/request.json')
report_path = Path('runs/final-evaluation/report.json')
report = read(report_path)
print('Loaded final report; validating frozen identities.', flush=True)
players = ['base', 'base-fp32', 'trained', 'random', 'heuristic']
seeds = list(range(30000, 30200))
assert request['players'] == players
assert set(report['players']) == set(players)
for key, expected in [('mode', 'tournament'), ('final_test', True), ('max_pieces', 200), ('seeds', seeds)]:
    assert report[key] == request[key] == expected, key
for file, expected in request['source_hashes'].items():
    assert sha(file) == expected, file
assert report['provenance']['source_hashes'] == request['source_hashes']
assert report['provenance']['installed_versions'] == request['installed_versions']
checkpoint = Path(request['checkpoint'])
checkpoint_hashes = evaluator.checkpoint_hashes(checkpoint)
assert checkpoint_hashes == request['checkpoint_sha256'] == report['checkpoint_sha256']
selection_path = Path(request['selection_file'])
assert sha(selection_path) == request['selection_sha256']
assert read(selection_path) == report['selection']
args = SimpleNamespace(seeds=tuple(seeds), final_test=True, max_pieces=200, players=players,
                       selection_file=selection_path, checkpoint=checkpoint, max_length=4096)
evaluator.validate_selection(args, checkpoint_hashes)
evaluator.validate_neural_runtimes(report['players'])
episodes = report['episodes']
assert len(episodes) == 1000
indexed = {(e['player_id'], e['seed']): e for e in episodes}
assert len(indexed) == 1000
assert set(indexed) == {(p, s) for p in players for s in seeds}
metrics = ('lines', 'score', 'pieces')
summaries = {}
episode_hashes = {}
def summary(values):
    if not values:
        return dict(count=0, mean=None, median=None, p95=None)
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p95=sorted(values)[math.ceil(.95 * len(values)) - 1])
for player in players:
    metadata = read(Path('runs/final-evaluation') / player / 'player.json')
    assert metadata == report['players'][player]
    rows = []
    for seed in seeds:
        path = Path('runs/final-evaluation') / player / f'seed-{seed}.json'
        episode = read(path)
        assert episode == indexed[player, seed], str(path)
        assert episode['player']['revision'] == metadata['revision']
        assert episode['player']['runtime_config'] == metadata['runtime_config']
        assert episode['max_pieces'] == 200
        exporter.validated_player(episode, 0)
        if player in ('base', 'base-fp32', 'trained'):
            assert all(d['probabilities'] and 0 < d['input_tokens'] <= 4096 for d in episode['decisions'])
        episode_hashes[str(path)] = sha(path)
        rows.append(episode)
    primary = {m: summary([0 if e['errors'] else e['outcome'][m] for e in rows]) for m in metrics}
    observed = {m: summary([e['outcome'][m] for e in rows]) for m in metrics}
    failed = [e['seed'] for e in rows if e['errors']]
    cap_count = sum(e['outcome']['cap_hit'] for e in rows)
    events = [d for e in rows for d in e['decisions']]
    computed = dict(episodes=200, failed_episodes=len(failed), error_rate=len(failed)/200,
                    invalid_decisions=sum(er['kind']=='invalid_decision' for e in rows for er in e['errors']),
                    cap_hit_rate=cap_count/200, failure_adjusted=primary, observed_before_error=observed,
                    latency_seconds=summary([d['decision_seconds'] for d in events]),
                    input_tokens=summary([d['input_tokens'] for d in events if d.get('input_tokens') is not None]),
                    failed_seeds=failed)
    for comparison in ('trained_vs_base', 'trained_vs_heuristic', 'trained_vs_base_fp32', 'base_fp32_vs_base'):
        stored = report[comparison]['players'][player]
        for key, value in computed.items():
            assert stored[key] == value, (comparison, player, key)
    summaries[player] = {**computed, 'cap_count': cap_count}
    print(f'{player}: all 200 saved replays/observations/decisions verified; mean lines={primary["lines"]["mean"]}', flush=True)
comparisons = {}
for key, left, right in [('trained_vs_base','trained','base'), ('trained_vs_heuristic','trained','heuristic'),
                          ('trained_vs_base_fp32','trained','base-fp32'), ('base_fp32_vs_base','base-fp32','base')]:
    stored = report[key]
    assert stored['trained_id'] == left and stored['base_id'] == right
    assert stored['seeds'] == seeds and stored['matches_reserved_test_pool'] is True
    assert stored['max_pieces'] == 200 and stored['selection_performed'] is False
    assert stored['max_error_rate'] == 0
    comparisons[key] = {}
    for metric in metrics:
        def score(p, seed):
            e = indexed[p, seed]
            return 0 if e['errors'] else e['outcome'][metric]
        deltas = [score(left,s)-score(right,s) for s in seeds]
        rng = random.Random(2026)
        means = sorted(sum(rng.choice(deltas) for _ in seeds)/200 for _ in range(10000))
        def percentile(q):
            n = 9999*q
            low = int(n)
            return means[low] + (means[math.ceil(n)]-means[low])*(n-low)
        expected = dict(episodes=200, mean_difference=statistics.mean(deltas), ci95_lower=percentile(.025),
                        ci95_upper=percentile(.975), method='paired episode percentile bootstrap',
                        bootstrap_samples=10000, bootstrap_seed=2026)
        assert expected == stored['paired_trained_minus_base'][metric], (key, metric)
        comparisons[key][metric] = expected
    acceptable = not summaries[left]['failed_seeds'] and not summaries[right]['failed_seeds']
    assert stored['errors_acceptable'] == acceptable
    assert stored['positive_mean_lines_signal'] == (comparisons[key]['lines']['ci95_lower'] > 0 and acceptable)
    print(f'{key}: all three paired metrics and bootstrap intervals independently match.', flush=True)
result = dict(status='passed', audit_script_sha256=sha(__file__), elapsed_seconds=time.monotonic()-start,
              report_sha256=sha(report_path), request_sha256=sha('runs/final-evaluation/request.json'),
              selection_sha256=sha(selection_path), checkpoint_sha256=checkpoint_hashes,
              source_hashes=request['source_hashes'], episode_file_sha256=episode_hashes,
              episodes_verified=1000, summaries=summaries, comparisons=comparisons,
              method='Saved replay reconstruction; independent statistics and random.choice paired bootstrap. No new policy decisions/model loads.')
OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({k: result[k] for k in ('status','elapsed_seconds','report_sha256','selection_sha256','episodes_verified')}), flush=True)

```
