#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比 /compare_cache/hf 与 /compare_cache/vllm 目录下的中间结果 (.pt) 张量差异。

功能:
1. 自动匹配文件名 (默认只看后缀 .pt)。
2. 读取张量 (torch.save / torch.load 保存的对象: Tensor 或 能转成 Tensor 的 list/np)。
3. 计算统计信息: 形状/数据类型 是否一致, mean/max/median/95% 分位 |diff|, 相对误差 (对 hf 归一), 余弦相似度, MSE, PSNR, 多阈值覆盖率 (|diff|≤阈值的比例)。
4. 针对每个文件生成: 差值分布直方图, (若张量 <= 2 维或可压缩) heatmap 对比 (hf, vllm, diff)。
5. 对超大张量采用采样显示 (随机 + 前 K) 以避免 OOM。
6. 可选: 保存差值张量 (--save-diff)。
7. 小张量输出完整元素对照 (--full-table-limit)。
8. 生成 Markdown 汇总报告 (summary.md) + 全局阈值覆盖率。

用法示例:
python compare_tensor.py \
  --hf-dir ../compare_cache/hf \
  --vllm-dir ../compare_cache/vllm \
  --out-dir ../compare_cache/report \
  --topk 10 --sample 50000

若只想打印统计:
python compare_tensor.py --no-plot
"""
from __future__ import annotations

import argparse
import os
import sys
import math
import random
import csv
from pathlib import Path
from typing import Any, Dict, Tuple

import torch


def to_tensor(obj) -> torch.Tensor:
	if isinstance(obj, torch.Tensor):
		return obj
	try:
		return torch.as_tensor(obj)
	except Exception:
		raise TypeError(f"无法转换对象为Tensor: type={type(obj)}")


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
	a_f = a.float().view(-1)
	b_f = b.float().view(-1)
	denom = (a_f.norm() * b_f.norm()).item()
	if denom == 0:
		return float('nan')
	return (a_f @ b_f).item() / denom


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
	mse = torch.mean((a.float() - b.float()) ** 2).item()
	if mse == 0:
		return float('inf')
	# 动态范围: 使用两侧数据混合的 max - min
	data_max = torch.max(torch.stack([a.max(), b.max()])).item()
	data_min = torch.min(torch.stack([a.min(), b.min()])).item()
	rng = data_max - data_min if data_max > data_min else 1.0
	return 10 * math.log10((rng ** 2) / mse)


def format_num(x: float) -> str:
	if x is None:
		return "-"
	if math.isinf(x):
		return "inf"
	if math.isnan(x):
		return "nan"
	if abs(x) >= 1e3 or abs(x) < 1e-3:
		return f"{x:.3e}"
	return f"{x:.6f}".rstrip('0').rstrip('.')


def stats_for_pair(hf_t: torch.Tensor, vllm_t: torch.Tensor, rel_eps: float = 1e-12) -> Dict[str, Any]:
	same_shape = tuple(hf_t.shape) == tuple(vllm_t.shape)
	same_dtype = hf_t.dtype == vllm_t.dtype
	if not same_shape:
		return {
			'same_shape': False,
			'same_dtype': same_dtype,
			'shape_hf': tuple(hf_t.shape),
			'shape_vllm': tuple(vllm_t.shape),
		}

	diff = (hf_t - vllm_t).float()
	abs_diff = diff.abs().view(-1)
	hf_abs = hf_t.float().abs().view(-1)
	denom = hf_abs + rel_eps
	rel_diff = (abs_diff / denom).clamp(max=1e6)

	metrics = {
		'same_shape': True,
		'same_dtype': same_dtype,
		'shape': tuple(hf_t.shape),
		'numel': hf_t.numel(),
		'mean_abs': abs_diff.mean().item(),
		'max_abs': abs_diff.max().item(),
		'median_abs': abs_diff.median().item(),
		'p95_abs': torch.quantile(abs_diff, 0.95).item() if abs_diff.numel() > 1 else abs_diff.item(),
		'mean_rel': rel_diff.mean().item(),
		'max_rel': rel_diff.max().item(),
		'cosine': cosine_similarity(hf_t, vllm_t),
		'mse': torch.mean(diff ** 2).item(),
		'psnr': psnr(hf_t, vllm_t),
		'hf_min': hf_t.min().item(),
		'hf_max': hf_t.max().item(),
		'vllm_min': vllm_t.min().item(),
		'vllm_max': vllm_t.max().item(),
	}
	return metrics


def prepare_plot():
	try:
		import matplotlib
		matplotlib.use('Agg')  # non-GUI
		import matplotlib.pyplot as plt  # noqa: F401
		import seaborn  # noqa: F401
		return True
	except Exception as e:
		print(f"[警告] 无法导入绘图依赖: {e}")
		return False


def plot_hist(abs_diff: torch.Tensor, out_png: Path, title: str):
	import matplotlib.pyplot as plt
	import seaborn as sns
	arr = abs_diff.cpu().numpy().flatten()
	plt.figure(figsize=(6,4))
	sns.histplot(arr, bins=80, log_scale=(False, True))
	plt.xlabel('|diff|')
	plt.ylabel('count (log)')
	plt.title(title)
	plt.tight_layout()
	plt.savefig(out_png)
	plt.close()


def plot_heatmaps(hf_t: torch.Tensor, vllm_t: torch.Tensor, out_dir: Path, base_name: str):
	import matplotlib.pyplot as plt
	import seaborn as sns
	# 仅处理 1D/2D, 其他尝试降维
	def to2d(t: torch.Tensor):
		if t.dim() == 1:
			return t.unsqueeze(0)
		if t.dim() == 2:
			return t
		# 展平到 (N, M)
		flat = t.view(t.shape[0], -1)
		# 若行数过大, 采样前 64 行
		if flat.shape[0] > 64:
			flat = flat[:64]
		# 列过大, 只取前 128 列
		if flat.shape[1] > 128:
			flat = flat[:, :128]
		return flat

	hf2 = to2d(hf_t).float()
	vllm2 = to2d(vllm_t).float()
	diff2 = (hf2 - vllm2)
	vmax = max(hf2.abs().max().item(), vllm2.abs().max().item())

	plt.figure(figsize=(12,4))
	for i,(data,title) in enumerate([
		(hf2, 'HF'), (vllm2, 'vLLM'), (diff2, 'Diff')
	]):
		plt.subplot(1,3,i+1)
		if title == 'Diff':
			sns.heatmap(data, cmap='coolwarm', center=0)
		else:
			sns.heatmap(data, cmap='viridis', vmin=-vmax, vmax=vmax)
		plt.title(title)
	plt.tight_layout()
	plt.savefig(out_dir / f"{base_name}_heatmap.png")
	plt.close()


def sample_tensor(t: torch.Tensor, max_sample: int) -> torch.Tensor:
	if t.numel() <= max_sample:
		return t.view(-1)
	flat = t.view(-1)
	# 前K + 随机
	k = max_sample // 2
	front = flat[:k]
	remain = flat[k:]
	idx = torch.randperm(remain.numel())[: (max_sample - k)]
	sampled = torch.cat([front, remain[idx]])
	return sampled


def ensure_dir(p: Path):
	p.mkdir(parents=True, exist_ok=True)


def threshold_coverage(abs_diff: torch.Tensor, thresholds) -> Dict[str, float]:
	cov = {}
	n = abs_diff.numel()
	if n == 0:
		return {f"≤{t}": float('nan') for t in thresholds}
	for t in thresholds:
		cov[f"≤{t}"] = (abs_diff <= t).float().mean().item()
	return cov


def save_small_tensor_table(hf_t: torch.Tensor, vllm_t: torch.Tensor, out_csv: Path):
	flat_h = hf_t.view(-1).float()
	flat_v = vllm_t.view(-1).float()
	diff = (flat_h - flat_v).abs()
	with open(out_csv, 'w', newline='', encoding='utf-8') as f:
		w = csv.writer(f)
		w.writerow(['index', 'hf', 'vllm', 'abs_diff'])
		for i in range(flat_h.numel()):
			w.writerow([i, float(flat_h[i]), float(flat_v[i]), float(diff[i])])


def compare_dirs(hf_dir: Path, vllm_dir: Path, out_dir: Path, do_plot: bool, max_sample: int, topk: int, thresholds=(), save_diff=False, full_table_limit=0) -> Tuple[Dict[str, Dict[str, Any]], Path]:
	ensure_dir(out_dir)
	filenames = sorted({f for f in os.listdir(hf_dir) if f.endswith('.pt')})
	results: Dict[str, Dict[str, Any]] = {}
	plot_ready = prepare_plot() if do_plot else False

	global_abs = []
	for fname in filenames:
		hf_path = hf_dir / fname
		vllm_path = vllm_dir / fname
		if not vllm_path.exists():
			print(f"[缺失] vLLM 缺少文件: {fname}")
			results[fname] = {'missing_vllm': True}
			continue
		try:
			hf_obj = torch.load(hf_path, map_location='cpu')
			vllm_obj = torch.load(vllm_path, map_location='cpu')
			hf_t = to_tensor(hf_obj)
			vllm_t = to_tensor(vllm_obj)
		except Exception as e:
			print(f"[错误] 读取 {fname} 失败: {e}")
			results[fname] = {'error': str(e)}
			continue

		metric = stats_for_pair(hf_t, vllm_t)
		results[fname] = metric

		if metric.get('same_shape'):
			# 为大张量构建抽样的差异直方图
			if do_plot and plot_ready:
				file_out_dir = out_dir / Path(fname).stem
				ensure_dir(file_out_dir)
				diff_abs_all = (hf_t - vllm_t).abs().float().view(-1)
				sampled = sample_tensor(diff_abs_all, max_sample)
				try:
					plot_hist(sampled, file_out_dir / 'diff_hist.png', f'{fname} |diff| histogram (sampled)')
					plot_heatmaps(hf_t, vllm_t, file_out_dir, Path(fname).stem)
				except Exception as pe:
					print(f"[警告] 绘图失败 {fname}: {pe}")

			# 记录 topk 最大差异索引及值 + 阈值覆盖率/保存差值
			diff_all = (hf_t - vllm_t)
			diff_abs = diff_all.abs().view(-1)
			if thresholds:
				metric['threshold_cov'] = threshold_coverage(diff_abs, thresholds)
				global_abs.append(diff_abs)
			if save_diff:
				file_out_dir = out_dir / Path(fname).stem
				try:
					ensure_dir(file_out_dir)
					torch.save(diff_all, file_out_dir / 'diff_tensor.pt')
				except Exception as de:
					print(f"[警告] 保存差值张量失败 {fname}: {de}")
			if full_table_limit > 0 and hf_t.numel() <= full_table_limit:
				file_out_dir = out_dir / Path(fname).stem
				try:
					ensure_dir(file_out_dir)
					save_small_tensor_table(hf_t, vllm_t, file_out_dir / 'full_elements.csv')
				except Exception as ce:
					print(f"[警告] 保存完整元素表失败 {fname}: {ce}")

			# topK
			if diff_abs.numel() <= topk:
				top_indices = torch.arange(diff_abs.numel())
			else:
				top_vals, top_indices = torch.topk(diff_abs, topk)
			else_list = []
			flat_hf = hf_t.view(-1)
			flat_v = vllm_t.view(-1)
			for i in range(min(topk, top_indices.numel())):
				idx = top_indices[i].item()
				else_list.append({
					'index': idx,
					'hf': flat_hf[idx].item(),
					'vllm': flat_v[idx].item(),
					'abs_diff': diff_abs[idx].item(),
				})
			metric['top_diff'] = else_list

	# 全局阈值覆盖率
	global_cov = {}
	if thresholds and global_abs:
		all_abs = torch.cat(global_abs)
		global_cov = threshold_coverage(all_abs, thresholds)

	# 写汇总 markdown
	summary_path = out_dir / 'summary.md'
	with open(summary_path, 'w', encoding='utf-8') as f:
		f.write('# HF 与 vLLM 中间结果对比汇总\n\n')
		f.write('| 文件 | shape | numel | mean_abs | max_abs | p95_abs | mean_rel | max_rel | cosine | mse | psnr | 备注 |\n')
		f.write('|------|-------|-------|---------|---------|---------|----------|---------|--------|-----|------|------|\n')
		for fname, m in results.items():
			if 'error' in m:
				note = '读取失败'
				f.write(f"| {fname} | - | - | - | - | - | - | - | - | - | - | {note} |\n")
				continue
			if m.get('missing_vllm'):
				f.write(f"| {fname} | - | - | - | - | - | - | - | - | - | - | vLLM缺失 |\n")
				continue
			if not m.get('same_shape'):
				f.write(f"| {fname} | hf:{m.get('shape_hf')} / v:{m.get('shape_vllm')} | - | - | - | - | - | - | - | - | - | 形状不一致 |\n")
				continue
			f.write(
				'| {fname} | {shape} | {numel} | {mean_abs} | {max_abs} | {p95_abs} | {mean_rel} | {max_rel} | {cosine} | {mse} | {psnr} | {note} |\n'.format(
					fname=fname,
					shape=m.get('shape'),
					numel=m.get('numel'),
					mean_abs=format_num(m.get('mean_abs')),
					max_abs=format_num(m.get('max_abs')),
					p95_abs=format_num(m.get('p95_abs')),
					mean_rel=format_num(m.get('mean_rel')),
					max_rel=format_num(m.get('max_rel')),
					cosine=format_num(m.get('cosine')),
					mse=format_num(m.get('mse')),
					psnr=format_num(m.get('psnr')),
					note='' if m.get('same_dtype') else 'dtype不同'
				)
			)
		if thresholds:
			f.write('\n## 阈值覆盖率 (单文件)\n')
			header = '| 文件 | ' + ' | '.join([f'≤{t}' for t in thresholds]) + ' |\n'
			f.write(header)
			f.write('|------|' + '|'.join(['-------'] * len(thresholds)) + '|\n')
			for fname, m in results.items():
				if not m.get('same_shape') or 'threshold_cov' not in m:
					continue
				cov_row = [f"{100*m['threshold_cov'][f'≤{t}']:.2f}%" for t in thresholds]
				f.write(f"| {fname} | " + ' | '.join(cov_row) + ' |\n')
			if global_cov:
				f.write('\n### 全局覆盖率 (合并所有文件)\n')
				for t in thresholds:
					f.write(f"- ≤{t}: {100*global_cov[f'≤{t}']:.4f}%\n")

		f.write('\n\n## Top 差异元素 (仅记录各文件内前若干)\n')
		for fname, m in results.items():
			if not m.get('same_shape') or 'top_diff' not in m:
				continue
			f.write(f"\n### {fname}\n\n")
			f.write('| 排名 | index(flat) | hf | vllm | abs_diff |\n')
			f.write('|------|-------------|----|------|----------|\n')
			for rank, item in enumerate(m['top_diff'], 1):
				f.write(f"| {rank} | {item['index']} | {format_num(item['hf'])} | {format_num(item['vllm'])} | {format_num(item['abs_diff'])} |\n")
	return results, summary_path


def main():
	parser = argparse.ArgumentParser(description='对比 HF 与 vLLM 中间结果张量差异')
	parser.add_argument('--hf-dir', type=Path, default=Path('../compare_cache/hf'))
	parser.add_argument('--vllm-dir', type=Path, default=Path('../compare_cache/vllm'))
	parser.add_argument('--out-dir', type=Path, default=Path('../compare_cache/report'))
	parser.add_argument('--no-plot', action='store_true', help='不生成图片')
	parser.add_argument('--sample', type=int, default=20000, help='直方图最大采样元素数')
	parser.add_argument('--topk', type=int, default=10, help='记录每个文件最大差异元素个数')
	parser.add_argument('--thresholds', type=str, default='1e-8,1e-7,1e-6,1e-5,1e-4,1e-3', help='逗号分隔阈值集合, 用于统计 |diff| 覆盖率; 空字符串关闭')
	parser.add_argument('--save-diff', action='store_true', help='保存差值张量 diff_tensor.pt')
	parser.add_argument('--full-table-limit', type=int, default=0, help='若张量元素总数 ≤ 该值, 输出完整元素对照表 csv')
	args = parser.parse_args()

	if not args.hf_dir.exists():
		print(f"HF 目录不存在: {args.hf_dir}")
		sys.exit(1)
	if not args.vllm_dir.exists():
		print(f"vLLM 目录不存在: {args.vllm_dir}")
		sys.exit(1)

	print(f"开始对比: HF={args.hf_dir} vLLM={args.vllm_dir} 输出={args.out_dir}")
	thresholds = [] if args.thresholds.strip() == '' else [float(x) for x in args.thresholds.split(',') if x.strip()]
	results, summary = compare_dirs(
		args.hf_dir,
		args.vllm_dir,
		args.out_dir,
		do_plot=not args.no_plot,
		max_sample=args.sample,
		topk=args.topk,
		thresholds=thresholds,
		save_diff=args.save_diff,
		full_table_limit=args.full_table_limit,
	)

	print(f"完成. 汇总报告: {summary}")
	# 控制台简表
	for fname, m in results.items():
		if m.get('same_shape'):
			print(f"[{fname}] mean_abs={format_num(m['mean_abs'])} max_abs={format_num(m['max_abs'])} cosine={format_num(m['cosine'])}")
		else:
			print(f"[{fname}] 形状或读取异常: {m}")


if __name__ == '__main__':
	main()

