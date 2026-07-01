"""
@Time: 4/8/2023 下午1:48
@Author: Heng Cai
@FileName: run_screening.py
@Copyright: 2020-2023 CarbonSilicon.ai
@Description:
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp

from tqdm import tqdm

from RTMScore.utils import scoring, get_rtmscore_model
from src.utils.docking_inference_utils import read_ligands, docking
from src.utils.docking_utils import extract_carsidock_pocket, read_mol, extract_pocket
from src.utils.chem_utils import safe_remove_hs
from src.utils.utils import get_abs_path, get_carsidock_model
import pytorch_lightning as pl


def get_heavy_atom_positions(ligand_file):
    ligand = read_mol(ligand_file)
    if ligand is None:
        ligand = read_mol(ligand_file, sanitize=False)
        positions = ligand.GetConformer().GetPositions()
        atoms = np.array([a.GetSymbol() for a in ligand.GetAtoms()])
        positions = positions[atoms != 'H']
    else:
        if ligand_file.endswith('.sdf'):
            ligand = ligand[0]
        ligand = safe_remove_hs(ligand)
        positions = ligand.GetConformer().GetPositions()
    return positions


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device_index)
    DEVICE = torch.device(f'cuda')

    if args.cuda_convert:
        import pydock
        lbfgsbsrv = pydock.LBFGSBServer(args.num_threads, args.cuda_device_index)
        print('Using cuda to accelerate distance matrix to coordinate.')
    else:
        lbfgsbsrv = None

    model, ligand_dict, pocket_dict = get_carsidock_model(args.carsidock_ckpt_path, DEVICE)
    rtms_model = get_rtmscore_model(get_abs_path(args.rtms_ckpt_path))
    pocket_file = get_abs_path(args.pdb_file)
    ligand_file = get_abs_path(args.reflig)
    positions = get_heavy_atom_positions(ligand_file)
    carsidock_pocket, _ = extract_carsidock_pocket(pocket_file, ligand_file)
    rtms_pocket = extract_pocket(pocket_file, positions, distance=10, del_water=True)

    if args.ligands.endswith('.sdf'):
        ligands = read_mol(get_abs_path(args.ligands))
        data = ligands
    elif args.ligands.endswith('.txt'):
        with open(get_abs_path(args.ligands), 'r', encoding='utf8') as f:
            smiles = [line.strip() for line in f.readlines()]
        data = smiles
    else:
        assert ValueError('only support .sdf or .txt file.')

    input_basename = os.path.basename(args.ligands).split('.')[0]
    score_filename = f"{input_basename}_score.dat"

    #docked_mol = []
    #invalid_count = 0
    #for item in tqdm(data):
#        try:
#            init_mol_list = read_ligands(smiles=[item])[0] if type(item) is str else read_ligands([item])[0]
#            torch.cuda.empty_cache()
#            if args.output_dir:
#                output_path = get_abs_path(args.output_dir, f'{init_mol_list[0].GetProp("_Name")}.sdf')
#            else:
#                output_path = None
#            outputs = docking(model, carsidock_pocket, init_mol_list, ligand_dict, pocket_dict, device=DEVICE,
#                              output_path=output_path, num_threads=args.num_threads, lbfgsbsrv=lbfgsbsrv)
#            docked_mol.append(outputs['mol_list'][0])
#        except(IndexError, AttributeError) as e:
#            invalid_count += 1
#            continue
#    ids, scores = scoring(rtms_pocket, docked_mol, rtms_model)
#    if args.output_dir is not None:
#        df = pd.DataFrame(zip(ids, scores), columns=["#code_ligand_num", "score"])
#        df.to_csv(f"{get_abs_path(args.output_dir)}/{score_filename}", index=False, sep="\t")

    pending_mol = []
    invalid_count = 0
    all_ids = []
    all_scores = []
    data = list(data)

    # 生成动态文件名
    input_basename = os.path.basename(args.ligands).split('.')[0]
    score_filename = f"{input_basename}_score.dat"

    # 设置每批的大小。每满一批就 scoring 并释放这一批，避免 10k 分子全部留在显存/内存里。
    batch_size = 1000
    scored_batches = 0

    def score_and_save_batch(mol_batch, batch_label):
        ids_batch, scores_batch = scoring(rtms_pocket, mol_batch, rtms_model)
        batch_df = pd.DataFrame({"#code_ligand_num": ids_batch, "score": scores_batch})
        batch_file = f"{input_basename}_batch_{batch_label}_score.dat"
        batch_df.to_csv(
            os.path.join(get_abs_path(args.output_dir), batch_file),
            index=False,
            sep="\t"
        )
        all_ids.extend(ids_batch)
        all_scores.extend(scores_batch)

    # 分批处理
    for item in tqdm(data, desc="Processing"):
        try:
            init_mol_list = read_ligands(smiles=[item])[0] if type(item) is str else read_ligands([item])[0]
            torch.cuda.empty_cache()
            if args.output_dir:
                output_path = get_abs_path(args.output_dir, f'{init_mol_list[0].GetProp("_Name")}.sdf')
            else:
                output_path = None
            outputs = docking(
                model, carsidock_pocket, init_mol_list, ligand_dict, pocket_dict, device=DEVICE,
                output_path=output_path, num_threads=args.num_threads, lbfgsbsrv=lbfgsbsrv
            )
            pending_mol.append(outputs['mol_list'][0])
        except (IndexError, AttributeError, ValueError) as e:
            invalid_count += 1
            continue

        if len(pending_mol) >= batch_size:
            scored_batches += 1
            score_and_save_batch(pending_mol, scored_batches)
            pending_mol.clear()
            torch.cuda.empty_cache()

    # 处理剩余不足 batch_size 的分子
    if pending_mol:
        score_and_save_batch(pending_mol, "final")
        pending_mol.clear()
        torch.cuda.empty_cache()

    # 最终合并所有批次的结果
    if args.output_dir and (all_ids and all_scores):
        final_df = pd.DataFrame({
            "#code_ligand_num": all_ids,
            "score": all_scores
        })

        output_path = os.path.join(get_abs_path(args.output_dir), score_filename)
        final_df.to_csv(output_path, index=False, sep="\t")
        print(f"Results saved to: {output_path}")

    print(f"Total invalid molecules skipped: {invalid_count}")

if __name__ == '__main__':
    pl.seed_everything(42)
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_file', default="example_data/ace_p.pdb",
                        help='protein file name')
    parser.add_argument('--reflig', default='example_data/ace_l.sdf',
                        help='the reference ligand to determine the pocket')
    parser.add_argument('--ligands', default='example_data/ace_decoys.sdf',
                        help='ligand decoys.')
    parser.add_argument('--output_dir', default='outputs/screening')
    parser.add_argument('--carsidock_ckpt_path', default='checkpoints/carsidock_230731.ckpt')
    parser.add_argument('--rtms_ckpt_path', default='checkpoints/rtmscore_model1.pth')
    parser.add_argument('--num_conformer', default=3, type=int,
                        help='number of initial conformer, resulting in num_conformer * num_conformer docking conformations.')
    parser.add_argument('--num_threads', default=1, type=int, help='recommend 1')
    parser.add_argument('--cuda_convert', action='store_true',
                        help='use cuda to accelerate distance matrix to coordinate.')
    parser.add_argument('--cuda_device_index', default=0, type=int, help="cuda device index")
    args = parser.parse_args()
    main(args)