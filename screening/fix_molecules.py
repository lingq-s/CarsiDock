#!/usr/bin/env python3
import os
import argparse
from rdkit import Chem
from rdkit.Chem import rdchem

def fix_hydrazone_structure(mol):
    """修改CO=NN结构为CON=N+"""
    query = Chem.MolFromSmiles('CO=NN', sanitize=False)
    matches = mol.GetSubstructMatches(query)

    if not matches:
        return None

    rwmol = Chem.RWMol(mol)

    for match in matches:
        idx_O = match[1]
        idx_N1 = match[2]
        idx_N2 = match[3]

        bond = rwmol.GetBondBetweenAtoms(idx_O, idx_N1)
        if bond:
            bond.SetBondType(rdchem.BondType.SINGLE)

        bond = rwmol.GetBondBetweenAtoms(idx_N1, idx_N2)
        if bond:
            bond.SetBondType(rdchem.BondType.DOUBLE)

        atom = rwmol.GetAtomWithIdx(idx_N2)
        atom.SetFormalCharge(1)

    return rwmol.GetMol()

def process_file(input_path, output_path, mode='modified', failed_output=None):
    # 第一次扫描：获取初始统计
    supplier_init = Chem.SDMolSupplier(input_path, sanitize=True, removeHs=True)
    original_total = len(supplier_init)
    original_invalid = [idx+1 for idx, mol in enumerate(supplier_init) if mol is None]

    # 第二次处理：实际修复
    supplier = Chem.SDMolSupplier(input_path, sanitize=False, removeHs=True)
    main_writer = Chem.SDWriter(output_path) if mode in ['all', 'both'] else None
    mod_writer = Chem.SDWriter(os.path.splitext(output_path)[0] + '_modified.sdf') if mode in ['modified', 'both'] else None

    step1_fixed = 0
    step2_fixed = 0
    step3_fixed = 0
    remaining_mols = []
    output_count = 0
    modified_count = 0

    for idx, mol in enumerate(supplier):
        original_mol = Chem.Mol(mol) if mol else None
        # was_modified = False

        # 如果是初始合法分子
        if (idx+1) not in original_invalid:
            if mol is not None:
                if main_writer: main_writer.write(mol)
                output_count += 1
            continue

        if mol is None:
            remaining_mols.append((idx+1, "无法解析的分子"))
            continue

        # 第一步修复
        needs_fix = False
        edit_mol = Chem.RWMol(mol)
        for atom in edit_mol.GetAtoms():
            if (atom.GetSymbol() == 'N' and 
                atom.GetFormalCharge() == 0 and 
                atom.GetExplicitValence() > 3):
                atom.SetFormalCharge(1)
                needs_fix = True

        if needs_fix:
            mol = edit_mol.GetMol()
            try:
                Chem.SanitizeMol(mol)
                if main_writer: main_writer.write(mol)
                if mod_writer: mod_writer.write(mol)
                step1_fixed += 1
                output_count += 1
                modified_count += 1
                continue
            except Exception as e:
                pass

        # 第二步修复
        edit_mol = Chem.RWMol(mol)
        for atom in edit_mol.GetAtoms():
            if atom.GetSymbol() == 'N' and atom.GetFormalCharge() == 1 and atom.GetTotalValence() > 4:
                atom.SetFormalCharge(0)

        mol = edit_mol.GetMol()
        try:
            Chem.SanitizeMol(mol)
            if main_writer: main_writer.write(mol)
            if mod_writer: mod_writer.write(mol)
            step2_fixed += 1
            output_count += 1
            modified_count += 1
            continue
        except Exception as e:
            pass

        # 第三步修复
        modified_mol = fix_hydrazone_structure(mol)
        if modified_mol is not None:
            try:
                Chem.SanitizeMol(modified_mol)
                if main_writer: main_writer.write(modified_mol)
                if mod_writer: mod_writer.write(modified_mol)
                step3_fixed += 1
                output_count += 1
                modified_count += 1
                continue
            except Exception as e:
                pass

        # 如果所有修复都失败
        try:
            Chem.SanitizeMol(mol)
            remaining_mols.append((idx+1, "所有修复尝试均失败"))
        except Exception as e:
            remaining_mols.append((idx+1, str(e)))   

    # 关闭写入器
    if main_writer: main_writer.close()
    if mod_writer: mod_writer.close()

    # 输出未修复分子
    if remaining_mols and failed_output:
        with Chem.SDWriter(failed_output) as failed_writer:
            supplier_all = Chem.SDMolSupplier(input_path, sanitize=False)
            remaining_indices = {idx for idx, _ in remaining_mols}
            for idx, mol in enumerate(supplier_all):
                if (idx+1) in remaining_indices and mol is not None:
                    failed_writer.write(mol)

    print("\n===== 修复统计报告 =====")
    print(f"[输入文件] 总分子数: {original_total}")
    print(f"[输入文件] 初始无效分子: {len(original_invalid)} (索引: {original_invalid[:10]}{'...' if len(original_invalid)>10 else ''})")
    print(f"[修复结果] 第一步修复成功: {step1_fixed}")
    print(f"[修复结果] 第二步修复成功: {step2_fixed}")
    print(f"[修复结果] 第三步修复成功: {step3_fixed}")
    print(f"[修复结果] 修改总数: {modified_count}")
    print(f"[修复结果] 未能修复的分子: {len(remaining_mols)}")
    print("\n===== 输出文件 =====")
    print(f"主文件分子总数: {output_count} (含{output_count - modified_count}个初始有效分子)")
    print(f"实际修改分子数: {modified_count}")
    if mode in ['all', 'both']:
        print(f"主输出文件路径: {output_path}")
    if mode in ['modified', 'both']:
        mod_path = os.path.splitext(output_path)[0] + '_modified.sdf'
        print(f"修改分子文件路径: {mod_path}")
    if remaining_mols and failed_output:
        print(f"失败分子文件路径: {failed_output}")

def main():
    parser = argparse.ArgumentParser(description='分子修复工具（保持原有修复逻辑）')
    parser.add_argument('input', help='输入SDF文件路径')
    parser.add_argument('-o', '--output', help='输出文件路径（默认：输入文件名+_fixed.sdf）')
    parser.add_argument('-m', '--mode', choices=['all', 'modified', 'both'], default='modified',  help='输出模式：all=全部分子， modified=仅修改分子，both=两者都输出')
    parser.add_argument('-f', '--failed', help='未修复分子输出路径（默认：输入文件名+_failed.sdf）')
    args = parser.parse_args()
    
    # 设置默认输出路径
    if not args.output:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_fixed.sdf"
    
    if not args.failed:
        args.failed = os.path.splitext(args.output)[0] + "_failed.sdf"
    
    # 处理文件
    process_file(
        input_path=args.input,
        output_path=args.output,
        mode=args.mode,
        failed_output=args.failed
    )

if __name__ == '__main__':
    main()
