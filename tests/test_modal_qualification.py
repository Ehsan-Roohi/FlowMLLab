from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qa'))
from qualify_modal_cfd import assess,protocol


def analytic_rows():
    return [dict(nodes_per_diameter=d,Re=100,nx=20*d,ny=8*d,Mach=3**.5*.05,
        observation_time_D_over_U=100,statistics_start_D_over_U=45,
        statistical_convergence_pass=True,Cd_mean=1.3+1/d**2,St=.16+1/d**2,
        Lr_over_D=1.4+1/d**2) for d in (18,27,40)]


class QualificationTests(unittest.TestCase):
    def test_grid_success_never_promotes_research(self):
        result=assess(analytic_rows())
        self.assertTrue(result['grid_gate_pass']);self.assertFalse(result['research_ready'])
        for q in result['quantities'].values():self.assertAlmostEqual(q['observed_order'],2,places=7)

    def test_statistics_and_missing_grid_fail_closed(self):
        rows=analytic_rows();rows[0]['statistical_convergence_pass']=False
        self.assertFalse(assess(rows)['grid_gate_pass'])
        rows[0]['statistical_convergence_pass']='False'
        self.assertFalse(assess(rows)['grid_gate_pass'])
        with self.assertRaises(ValueError):assess(rows[:2])

    def test_incompatible_domain_rejected(self):
        rows=analytic_rows();rows[-1]['ny']*=2
        with self.assertRaises(ValueError):assess(rows)

    def test_nonmonotone_not_promoted(self):
        rows=analytic_rows();rows[1]['Cd_mean']=1
        self.assertFalse(assess(rows)['grid_gate_pass'])

    def test_protocol_freezes_original_settings(self):
        p=protocol();self.assertEqual(p['settings']['steps'],80000)
        self.assertEqual(p['settings']['statistics_start'],36000)
        self.assertEqual(p['gci_sequence'],[18,27,40]);self.assertFalse(p['research_ready'])
        self.assertEqual(len(p['solver_sha256']),64)


if __name__=='__main__':unittest.main()
