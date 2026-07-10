#!/usr/bin/env python3
"""
PATH batch converter (legacy).

For standard per-residue CSV compatible with other predictors, prefer::

    pip install -e .
    aggressor-parse path --results results.csv --fasta protein.fasta -o PATH_standard.csv

This script still exports extended tables (APR regions, summary statistics).
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings


@dataclass
class ProteinAPR:
    """Структура для хранения информации об амилоидогенном регионе."""
    protein_id: str
    start: int
    end: int
    length: int
    sequence: str
    mean_score: float
    max_score: float
    gatekeeper_N: str  # Amino acid at position start-1
    gatekeeper_C: str  # Amino acid at position end+1
    has_proline: bool
    

@dataclass
class ProteinStatistics:
    """Статистика обработки белка."""
    protein_id: str
    sequence_length: int
    coverage: float  # Fraction of residues with PATH predictions
    mean_score: float
    median_score: float
    max_score: float
    num_aprs: int  # Number of identified APRs
    total_apr_length: int  # Total length of APRs
    apr_density: float  # APR residues / total length
    

class FASTAParser:
    """Парсер FASTA файлов."""
    
    @staticmethod
    def parse(fasta_path: str) -> Dict[str, str]:
        """
        Парсит FASTA файл.
        
        Args:
            fasta_path: путь к FASTA файлу
        
        Returns:
            Dict: {protein_id: sequence}
        """
        sequences = {}
        current_id = None
        current_seq = []
        
        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences[current_id] = ''.join(current_seq)
                    
                    # Start new sequence
                    current_id = line[1:].split()[0]  # Take first word as ID
                    current_seq = []
                else:
                    current_seq.append(line)
            
            # Save last sequence
            if current_id is not None:
                sequences[current_id] = ''.join(current_seq)
        
        return sequences


class BatchPATHProcessor:
    """
    Batch processor для PATH predictions на множестве белков.
    
    ALGORITHM OVERVIEW:
    -------------------
    1. Load PATH results.csv and extract hexapeptide scores
    2. Normalize DOPE scores to [0,1] with INVERSION
    3. For each protein in FASTA:
       a. Generate sliding window hexapeptides
       b. Map to PATH scores (handle missing with default=0)
       c. Average overlapping windows → per-residue scores
       d. Apply threshold → binary classification
       e. Identify contiguous APRs (min_length=5)
       f. Calculate statistics and quality metrics
    4. Export structured tables for downstream analysis
    
    QUALITY CONTROL:
    ----------------
    - Warns if coverage < 50% (unreliable predictions)
    - Flags proteins with no PATH coverage (require alternative predictors)
    - Identifies boundary effects at N/C-termini
    - Validates sequence characters (only standard 20 amino acids)
    """
    
    def __init__(
        self,
        results_csv_path: str,
        threshold_percentile: float = 75.0,
        min_apr_length: int = 5,
        gap_tolerance: int = 2,
        verbose: bool = True
    ):
        """
        Args:
            results_csv_path: путь к PATH results.csv
            threshold_percentile: percentile для бинарной классификации (default: 75 = top 25%)
            min_apr_length: минимальная длина APR (default: 5 а.к.)
            gap_tolerance: максимальный gap внутри APR для слияния (default: 2 а.к.)
            verbose: печатать прогресс
        """
        self.results_path = results_csv_path
        self.threshold_percentile = threshold_percentile
        self.min_apr_length = min_apr_length
        self.gap_tolerance = gap_tolerance
        self.verbose = verbose
        
        # Data containers
        self.hexapeptide_scores = {}
        self.protein_sequences = {}
        self.per_residue_scores = {}
        self.binary_classifications = {}
        self.apr_regions = {}
        self.statistics = {}
        
        # Statistics for normalization
        self.dope_min = None
        self.dope_max = None
        self.global_threshold = None
        
        # Load PATH results
        self._load_path_results()
        
    def _load_path_results(self):
        """Загрузка и обработка PATH results.csv."""
        if self.verbose:
            print("="*80)
            print("LOADING PATH RESULTS")
            print("="*80)
        
        # Load data
        results = pd.read_csv(self.results_path)
        
        if self.verbose:
            print(f"✓ Loaded {len(results)} models for {results['seq'].nunique()} hexapeptides")
        
        # Extract best DOPE per hexapeptide (vectorized groupby)
        hexapeptide_dope = results.groupby("seq", sort=False)["dope"].min()

        # Calculate normalization parameters
        self.dope_min = results["dope"].min()
        self.dope_max = results["dope"].max()
        span = float(self.dope_max - self.dope_min)

        if self.verbose:
            print(f"\nDOPE range: [{self.dope_min:.2f}, {self.dope_max:.2f}]")

        # Normalize with INVERSION
        if span == 0.0:
            self.hexapeptide_scores = {seq: 0.0 for seq in hexapeptide_dope.index}
        else:
            self.hexapeptide_scores = {
                seq: (self.dope_max - dope) / span
                for seq, dope in hexapeptide_dope.items()
            }

        # Calculate global threshold
        scores_array = np.fromiter(self.hexapeptide_scores.values(), dtype=float)
        self.global_threshold = np.percentile(scores_array, self.threshold_percentile)
        
        if self.verbose:
            print(f"✓ Normalized {len(self.hexapeptide_scores)} hexapeptide scores")
            print(f"✓ Global threshold ({self.threshold_percentile}th percentile): {self.global_threshold:.4f}")
            print()
    
    def load_proteins(self, fasta_path: str):
        """
        Загрузка белковых последовательностей из FASTA.
        
        Args:
            fasta_path: путь к FASTA файлу
        """
        if self.verbose:
            print("="*80)
            print("LOADING PROTEIN SEQUENCES")
            print("="*80)
        
        self.protein_sequences = FASTAParser.parse(fasta_path)
        
        if self.verbose:
            print(f"✓ Loaded {len(self.protein_sequences)} protein sequences")
            for protein_id, seq in list(self.protein_sequences.items())[:5]:
                print(f"  - {protein_id}: {len(seq)} residues")
            if len(self.protein_sequences) > 5:
                print(f"  ... and {len(self.protein_sequences) - 5} more")
            print()
    
    def process_all_proteins(self):
        """Обработка всех загруженных белков."""
        if not self.protein_sequences:
            raise ValueError("No proteins loaded. Call load_proteins() first.")
        
        if self.verbose:
            print("="*80)
            print(f"PROCESSING {len(self.protein_sequences)} PROTEINS")
            print("="*80)
            print()
        
        for i, (protein_id, sequence) in enumerate(self.protein_sequences.items(), 1):
            if self.verbose:
                print(f"[{i}/{len(self.protein_sequences)}] Processing {protein_id}...")
            
            # Process protein
            per_residue, binary, aprs, stats = self._process_single_protein(
                protein_id, sequence
            )
            
            # Store results
            self.per_residue_scores[protein_id] = per_residue
            self.binary_classifications[protein_id] = binary
            self.apr_regions[protein_id] = aprs
            self.statistics[protein_id] = stats
            
            if self.verbose:
                print(f"  Coverage: {stats.coverage*100:.1f}% | "
                      f"APRs: {stats.num_aprs} | "
                      f"Mean score: {stats.mean_score:.3f}")
                
                # Warning for low coverage
                if stats.coverage < 0.5:
                    warnings.warn(
                        f"Low PATH coverage ({stats.coverage*100:.1f}%) for {protein_id}. "
                        "Consider using alternative predictors.",
                        UserWarning
                    )
        
        if self.verbose:
            print("\n✓ All proteins processed")
            print()
    
    def _process_single_protein(
        self,
        protein_id: str,
        sequence: str
    ) -> Tuple[np.ndarray, np.ndarray, List[ProteinAPR], ProteinStatistics]:
        """
        Обработка одного белка.
        
        Returns:
            Tuple: (per_residue_scores, binary_classification, apr_regions, statistics)
        """
        seq_len = len(sequence)
        score_sum = np.zeros(seq_len, dtype=float)
        count = np.zeros(seq_len, dtype=int)
        
        # Sliding window
        window = 6
        for i in range(seq_len - window + 1):
            hexapeptide = sequence[i:i+window]
            
            # Skip if contains invalid characters
            if not all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in hexapeptide):
                continue
            
            if hexapeptide in self.hexapeptide_scores:
                score = self.hexapeptide_scores[hexapeptide]
                score_sum[i:i+window] += score
                count[i:i+window] += 1
        
        # Calculate per-residue scores
        per_residue = np.zeros(seq_len)
        mask = count > 0
        per_residue[mask] = score_sum[mask] / count[mask]
        
        # Binary classification
        binary = (per_residue >= self.global_threshold).astype(int)
        
        # Identify APR regions
        aprs = self._identify_apr_regions(protein_id, sequence, per_residue, binary)
        
        # Calculate statistics
        coverage = mask.sum() / seq_len if seq_len > 0 else 0.0
        covered_scores = per_residue[mask]
        
        stats = ProteinStatistics(
            protein_id=protein_id,
            sequence_length=seq_len,
            coverage=coverage,
            mean_score=float(covered_scores.mean()) if len(covered_scores) > 0 else 0.0,
            median_score=float(np.median(covered_scores)) if len(covered_scores) > 0 else 0.0,
            max_score=float(covered_scores.max()) if len(covered_scores) > 0 else 0.0,
            num_aprs=len(aprs),
            total_apr_length=sum(apr.length for apr in aprs),
            apr_density=sum(apr.length for apr in aprs) / seq_len if seq_len > 0 else 0.0
        )
        
        return per_residue, binary, aprs, stats
    
    def _identify_apr_regions(
        self,
        protein_id: str,
        sequence: str,
        per_residue: np.ndarray,
        binary: np.ndarray
    ) -> List[ProteinAPR]:
        """
        Идентификация непрерывных амилоидогенных регионов (APRs).
        
        BIOLOGICAL RATIONALE:
        ---------------------
        - Минимальная длина: 5-6 а.к. (критический nucleus для фибриллизации)
        - Gap tolerance: 1-2 а.к. (допускает beta-breakers внутри APR)
        - Gatekeeper analysis: проверка фланкирующих позиций на P/K/R/E/D
        
        Returns:
            List[ProteinAPR]: список идентифицированных APRs
        """
        aprs = []
        seq_len = len(sequence)
        
        # Find contiguous regions with binary=1
        in_region = False
        start = 0
        
        for i in range(seq_len):
            if binary[i] == 1 and not in_region:
                start = i
                in_region = True
            elif binary[i] == 0 and in_region:
                # Check if we should close the region
                if i - start >= self.min_apr_length:
                    apr = self._create_apr_object(
                        protein_id, sequence, per_residue, start, i
                    )
                    aprs.append(apr)
                in_region = False
        
        # Handle last region
        if in_region and seq_len - start >= self.min_apr_length:
            apr = self._create_apr_object(
                protein_id, sequence, per_residue, start, seq_len
            )
            aprs.append(apr)
        
        # Merge close APRs (within gap_tolerance)
        if self.gap_tolerance > 0:
            aprs = self._merge_close_aprs(aprs, sequence, per_residue)
        
        return aprs
    
    def _create_apr_object(
        self,
        protein_id: str,
        sequence: str,
        per_residue: np.ndarray,
        start: int,
        end: int
    ) -> ProteinAPR:
        """Создание объекта ProteinAPR с метаданными."""
        apr_seq = sequence[start:end]
        apr_scores = per_residue[start:end]
        
        # Identify gatekeepers
        gatekeeper_N = sequence[start-1] if start > 0 else '-'
        gatekeeper_C = sequence[end] if end < len(sequence) else '-'
        
        # Check for proline
        has_proline = 'P' in apr_seq
        
        return ProteinAPR(
            protein_id=protein_id,
            start=start,
            end=end,
            length=end - start,
            sequence=apr_seq,
            mean_score=float(apr_scores.mean()),
            max_score=float(apr_scores.max()),
            gatekeeper_N=gatekeeper_N,
            gatekeeper_C=gatekeeper_C,
            has_proline=has_proline
        )
    
    def _merge_close_aprs(
        self,
        aprs: List[ProteinAPR],
        sequence: str,
        per_residue: np.ndarray
    ) -> List[ProteinAPR]:
        """
        Слияние близких APRs (separated by ≤ gap_tolerance residues).
        
        BIOLOGICAL JUSTIFICATION:
        -------------------------
        Короткие gaps (<3 а.к.) между APRs часто содержат charged или polar
        residues, которые не предотвращают фибриллизацию но могут быть
        ошибочно классифицированы как non-amyloidogenic из-за локального
        снижения hydrophobicity.
        """
        if len(aprs) <= 1:
            return aprs
        
        merged = []
        current = aprs[0]
        
        for next_apr in aprs[1:]:
            gap = next_apr.start - current.end
            
            if gap <= self.gap_tolerance:
                # Merge: extend current APR to include next
                merged_start = current.start
                merged_end = next_apr.end
                current = self._create_apr_object(
                    current.protein_id, sequence, per_residue,
                    merged_start, merged_end
                )
            else:
                # Gap too large: save current and start new
                merged.append(current)
                current = next_apr
        
        # Add last APR
        merged.append(current)
        
        return merged
    
    def export_results(self, output_dir: str):
        """
        Экспорт результатов в структурированные таблицы.
        
        Генерирует 4 файла:
        1. per_residue_scores.csv: детальные скоры
        2. binary_classification.csv: бинарные предсказания
        3. apr_regions.csv: идентифицированные APRs
        4. summary_statistics.csv: статистика по белкам
        
        Args:
            output_dir: директория для сохранения файлов
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print("="*80)
            print("EXPORTING RESULTS")
            print("="*80)
        
        # 1. Per-residue scores (long format)
        per_residue_rows = []
        for protein_id, scores in self.per_residue_scores.items():
            sequence = self.protein_sequences[protein_id]
            for position, (aa, score) in enumerate(zip(sequence, scores)):
                per_residue_rows.append({
                    'protein_id': protein_id,
                    'position': position + 1,  # 1-indexed
                    'amino_acid': aa,
                    'path_score': score,
                    'has_prediction': score > 0
                })
        
        df_scores = pd.DataFrame(per_residue_rows)
        scores_file = output_path / 'per_residue_scores.csv'
        df_scores.to_csv(scores_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported per-residue scores: {scores_file}")
        
        # 2. Binary classification (long format)
        binary_rows = []
        for protein_id, binary in self.binary_classifications.items():
            sequence = self.protein_sequences[protein_id]
            for position, (aa, pred) in enumerate(zip(sequence, binary)):
                binary_rows.append({
                    'protein_id': protein_id,
                    'position': position + 1,
                    'amino_acid': aa,
                    'is_amyloidogenic': pred
                })
        
        df_binary = pd.DataFrame(binary_rows)
        binary_file = output_path / 'binary_classification.csv'
        df_binary.to_csv(binary_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported binary classification: {binary_file}")
        
        # 3. APR regions
        apr_rows = []
        for protein_id, aprs in self.apr_regions.items():
            for apr in aprs:
                apr_dict = asdict(apr)
                apr_rows.append(apr_dict)
        
        df_aprs = pd.DataFrame(apr_rows)
        aprs_file = output_path / 'apr_regions.csv'
        df_aprs.to_csv(aprs_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported APR regions: {aprs_file}")
        
        # 4. Summary statistics
        stats_rows = [asdict(stats) for stats in self.statistics.values()]
        df_stats = pd.DataFrame(stats_rows)
        stats_file = output_path / 'summary_statistics.csv'
        df_stats.to_csv(stats_file, index=False)
        
        if self.verbose:
            print(f"✓ Exported summary statistics: {stats_file}")
            print()
            
            # Print summary
            print("="*80)
            print("PROCESSING SUMMARY")
            print("="*80)
            print(f"Total proteins processed: {len(self.protein_sequences)}")
            print(f"Total residues analyzed: {df_scores.shape[0]}")
            print(f"Residues with PATH predictions: {df_scores['has_prediction'].sum()}")
            print(f"Amyloidogenic residues identified: {df_binary['is_amyloidogenic'].sum()}")
            print(f"Total APRs identified: {len(df_aprs)}")
            print(f"\nMean coverage across proteins: {df_stats['coverage'].mean()*100:.1f}%")
            print(f"Proteins with coverage < 50%: {(df_stats['coverage'] < 0.5).sum()}")
            print("="*80)


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Batch PATH score converter for multiple proteins'
    )
    parser.add_argument(
        '--results',
        required=True,
        help='Path to PATH results.csv'
    )
    parser.add_argument(
        '--fasta',
        required=True,
        help='Path to FASTA file with protein sequences'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=75.0,
        help='Percentile threshold for binary classification (default: 75)'
    )
    parser.add_argument(
        '--min-apr-length',
        type=int,
        default=5,
        help='Minimum APR length (default: 5)'
    )
    parser.add_argument(
        '--gap-tolerance',
        type=int,
        default=2,
        help='Gap tolerance for merging APRs (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = BatchPATHProcessor(
        results_csv_path=args.results,
        threshold_percentile=args.threshold,
        min_apr_length=args.min_apr_length,
        gap_tolerance=args.gap_tolerance,
        verbose=True
    )
    
    # Load proteins
    processor.load_proteins(args.fasta)
    
    # Process all proteins
    processor.process_all_proteins()
    
    # Export results
    processor.export_results(args.output)


if __name__ == '__main__':
    main()