"""CPU interface example with artificial geometry; not an accuracy evaluation."""
import argparse
from pathlib import Path
import torch
from model_full import TASKS, load_model


def synthetic_input(task, chunked=False):
    torch.manual_seed(42)
    pos = torch.tensor([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.], [6., 0., 0.]])
    g = dict(
        nregions=4, patches=8, radius=2., ncoarse=2,
        sf=torch.randn(8, 4), pf=torch.randn(32, 4),
        pf_directional=torch.randn(32, 4),
        point_owner=torch.arange(8).repeat_interleave(4),
        point_weight=torch.ones(32), patch_mass=torch.full((8,), 4.),
        sr=torch.arange(4).repeat_interleave(2), sn=torch.arange(8),
        sw=torch.full((8,), .5), pr=torch.arange(4).repeat_interleave(2),
        pj=torch.arange(8), pw=torch.ones(8), phi=torch.randn(8, 5),
        graph_mass=torch.ones(4), surface_mass=torch.full((4,), 8.),
        has_g=torch.ones(4, dtype=torch.bool), has_p=torch.ones(4, dtype=torch.bool),
        pos=pos, s=pos.repeat_interleave(2, 0) + torch.randn(8, 3) * .1,
        length=torch.full((4,), 2.),
        adjacency=torch.tensor([[0, 1], [1, 2], [2, 3]]),
        coarse_owner=torch.tensor([0, 0, 1, 1]), query=pos[0],
        query_xyz=pos[:1] if task == 'ted' else pos[:3],
        pair_count=3, bridge_row=torch.tensor([0, 0, 1, 1, 2, 2]),
        bridge_col=torch.tensor([0, 1, 1, 2, 2, 3]), bridge_weight=torch.full((6,), .5),
        pair_lookup=torch.tensor([[0, 1, 2], [1, 0, 3], [2, 3, 0]]),
    )
    if chunked:
        for k in ('pr', 'pj', 'pw'):
            g[k] = g[k].repeat(4097)
        g['phi'] = g['phi'].repeat(4097, 1)
        g['surface_mass'] *= 4097
    return g


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=TASKS, default='h01_local')
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path = Path(__file__).resolve().parent / 'checkpoints' / a.task / 'final.pt'
    model = load_model(a.task, path)
    with torch.no_grad():
        z = model(synthetic_input(a.task))
    print('Task:', a.task, 'parameters:', sum(p.numel() for p in model.parameters()))
    print('Output shape:', tuple(z.shape), 'finite:', bool(torch.isfinite(z).all()))
    print('Synthetic input only. Output has no scientific performance meaning.')


if __name__ == '__main__':
    main()
