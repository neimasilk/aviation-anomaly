"""
Sequential SMOTE for Imbalanced CVR Data

Implements SMOTE (Synthetic Minority Oversampling Technique) for sequential data
by operating in BERT embedding space while maintaining temporal structure.

References:
- Chawla et al. (2002): SMOTE: Synthetic Minority Over-sampling Technique
- Extension for sequences: Interpolate in embedding space
"""
import random
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


class SequentialSMOTE:
    """
    SMOTE augmentation for sequential utterance data.
    
    Operates in BERT embedding space to generate synthetic sequences
    that maintain linguistic and temporal properties.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        k_neighbors: int = 5,
        random_seed: int = 42,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.k_neighbors = k_neighbors
        self.device = device
        self.random_seed = random_seed
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Load BERT for embedding extraction
        console = None
        try:
            from rich.console import Console
            console = Console()
            console.print(f"[yellow]Loading BERT encoder for SMOTE...[/yellow]")
        except:
            print("Loading BERT encoder for SMOTE...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.encoder.eval()
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        if console:
            console.print(f"[green]BERT encoder loaded[/green]")
    
    @torch.no_grad()
    def extract_sequence_embedding(
        self,
        utterances: List[str],
        max_length: int = 128,
    ) -> np.ndarray:
        """
        Extract embedding for a sequence of utterances.
        
        Returns mean-pooled BERT embedding of all utterances.
        """
        if not utterances:
            return np.zeros(self.encoder.config.hidden_size)
        
        # Tokenize all utterances
        encoded = self.tokenizer(
            utterances,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        
        # Get BERT embeddings
        outputs = self.encoder(**encoded)
        
        # Mean pool over tokens and utterances
        # Shape: (num_utterances, seq_len, hidden)
        embeddings = outputs.last_hidden_state
        
        # Mean over tokens (attention mask weighted)
        attention_mask = encoded['attention_mask'].unsqueeze(-1)
        sum_embeddings = (embeddings * attention_mask).sum(dim=1)
        mean_embeddings = sum_embeddings / attention_mask.sum(dim=1).clamp(min=1)
        
        # Mean over utterances
        sequence_embedding = mean_embeddings.mean(dim=0).cpu().numpy()
        
        return sequence_embedding
    
    def find_k_nearest_neighbors(
        self,
        target: np.ndarray,
        candidates: List[np.ndarray],
        k: int,
    ) -> List[int]:
        """Find k nearest neighbors using cosine similarity."""
        # Normalize for cosine similarity
        target_norm = target / (np.linalg.norm(target) + 1e-8)
        
        similarities = []
        for i, candidate in enumerate(candidates):
            candidate_norm = candidate / (np.linalg.norm(candidate) + 1e-8)
            sim = np.dot(target_norm, candidate_norm)
            similarities.append((sim, i))
        
        # Sort by similarity (descending) and get top k
        similarities.sort(reverse=True)
        return [idx for _, idx in similarities[:k]]
    
    def generate_synthetic_sample(
        self,
        sample1: Dict,
        sample2: Dict,
        alpha: Optional[float] = None,
    ) -> Dict:
        """
        Generate synthetic sample by interpolating two samples.
        
        Interpolation happens at the utterance level for corresponding positions.
        """
        if alpha is None:
            alpha = random.random()
        
        utterances1 = sample1['utterances']
        utterances2 = sample2['utterances']
        
        # Match sequence lengths (pad or truncate)
        max_len = max(len(utterances1), len(utterances2))
        
        # For simplicity, duplicate shorter sequence
        if len(utterances1) < max_len:
            utterances1 = utterances1 * (max_len // len(utterances1) + 1)
            utterances1 = utterances1[:max_len]
        if len(utterances2) < max_len:
            utterances2 = utterances2 * (max_len // len(utterances2) + 1)
            utterances2 = utterances2[:max_len]
        
        # Generate synthetic utterances (choose one randomly)
        # Alternative: Could use paraphrasing here
        synthetic_utterances = []
        for u1, u2 in zip(utterances1, utterances2):
            if random.random() < alpha:
                synthetic_utterances.append(u1)
            else:
                synthetic_utterances.append(u2)
        
        # Use the label of the primary sample
        synthetic_label = sample1['label']
        
        return {
            'case_id': f"synthetic_{random.randint(10000, 99999)}",
            'utterances': synthetic_utterances,
            'label': synthetic_label,
            'is_synthetic': True,
        }
    
    def fit_resample(
        self,
        sequences: List[Dict],
        labels: List[str],
        sampling_strategy: Dict[str, float],
    ) -> Tuple[List[Dict], List[str]]:
        """
        Apply SMOTE to balance classes according to sampling_strategy.
        
        Args:
            sequences: List of sequence dicts with 'utterances' and 'label'
            labels: List of sequence labels
            sampling_strategy: Dict mapping class name to target ratio (relative to majority)
                e.g., {'NORMAL': 1.0, 'CRITICAL': 0.5} means CRITICAL should be 50% of NORMAL
        
        Returns:
            Augmented sequences and labels
        """
        from collections import Counter
        
        print(f"Original distribution: {Counter(labels)}")
        
        # Group sequences by label
        class_sequences = {}
        for seq, label in zip(sequences, labels):
            if label not in class_sequences:
                class_sequences[label] = []
            class_sequences[label].append(seq)
        
        # Find majority class size
        majority_size = max(len(seqs) for seqs in class_sequences.values())
        
        # Calculate target sizes
        target_sizes = {}
        for label, ratio in sampling_strategy.items():
            if label in class_sequences:
                target_sizes[label] = int(majority_size * ratio)
        
        print(f"Target sizes: {target_sizes}")
        
        # Result containers
        resampled_sequences = []
        resampled_labels = []
        
        # Process each class
        for label, target_size in target_sizes.items():
            current_sequences = class_sequences.get(label, [])
            current_size = len(current_sequences)
            
            # Add original sequences
            resampled_sequences.extend(current_sequences)
            resampled_labels.extend([label] * current_size)
            
            if target_size <= current_size:
                # Undersample
                continue
            
            # Need to generate synthetic samples
            n_synthetic = target_size - current_size
            print(f"Generating {n_synthetic} synthetic samples for {label}...")
            
            # Extract embeddings for minority class
            print(f"  Extracting embeddings for {current_size} sequences...")
            embeddings = []
            for seq in tqdm(current_sequences, desc="Extracting"):
                emb = self.extract_sequence_embedding(seq['utterances'])
                embeddings.append(emb)
            
            # Generate synthetic samples
            synthetic_count = 0
            attempts = 0
            max_attempts = n_synthetic * 10
            
            while synthetic_count < n_synthetic and attempts < max_attempts:
                attempts += 1
                
                # Pick a random sample
                idx = random.randint(0, len(current_sequences) - 1)
                sample = current_sequences[idx]
                sample_emb = embeddings[idx]
                
                # Find k nearest neighbors
                neighbor_indices = self.find_k_nearest_neighbors(
                    sample_emb, embeddings, self.k_neighbors
                )
                
                # Pick a random neighbor
                neighbor_idx = random.choice(neighbor_indices)
                if neighbor_idx == idx and len(neighbor_indices) > 1:
                    # Avoid self-interpolation
                    neighbor_idx = random.choice([i for i in neighbor_indices if i != idx])
                
                neighbor = current_sequences[neighbor_idx]
                
                # Generate synthetic sample
                synthetic = self.generate_synthetic_sample(sample, neighbor)
                
                resampled_sequences.append(synthetic)
                resampled_labels.append(label)
                synthetic_count += 1
            
            print(f"  Generated {synthetic_count} synthetic samples")
        
        print(f"Final distribution: {Counter(resampled_labels)}")
        
        return resampled_sequences, resampled_labels


def apply_smote_to_dataset(
    df,
    label_column: str = "label",
    text_column: str = "cvr_message",
    case_id_column: str = "case_id",
    sampling_strategy: Optional[Dict[str, float]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> List[Dict]:
    """
    Apply SMOTE augmentation to a CVR dataset.
    
    Args:
        df: DataFrame with CVR data
        label_column: Name of label column
        text_column: Name of text column
        case_id_column: Name of case ID column
        sampling_strategy: Target ratios for each class
        device: Device for BERT encoding
    
    Returns:
        List of augmented sequences
    """
    if sampling_strategy is None:
        # Default: Balance all classes to match majority
        label_counts = df[label_column].value_counts()
        majority_count = label_counts.max()
        sampling_strategy = {
            label: count / majority_count
            for label, count in label_counts.items()
        }
    
    # Group by case to create sequences
    sequences = []
    labels = []
    
    for case_id, group in df.groupby(case_id_column):
        group = group.sort_values('turn_number') if 'turn_number' in group.columns else group
        
        utterances = []
        case_labels = []
        
        for _, row in group.iterrows():
            if pd.notna(row[text_column]):
                utterances.append(str(row[text_column]))
                case_labels.append(row[label_column])
        
        if utterances:
            # Use majority label for the sequence
            from collections import Counter
            majority_label = Counter(case_labels).most_common(1)[0][0]
            
            sequences.append({
                'case_id': case_id,
                'utterances': utterances,
                'label': majority_label,
                'is_synthetic': False,
            })
            labels.append(majority_label)
    
    # Apply SMOTE
    smote = SequentialSMOTE(device=device)
    augmented_sequences, augmented_labels = smote.fit_resample(
        sequences, labels, sampling_strategy
    )
    
    return augmented_sequences


if __name__ == "__main__":
    # Test SMOTE
    import pandas as pd
    
    print("Testing Sequential SMOTE...")
    
    # Create dummy data
    dummy_data = []
    for i in range(100):
        label = "NORMAL" if i < 65 else "EARLY_WARNING" if i < 85 else "ELEVATED" if i < 95 else "CRITICAL"
        dummy_data.append({
            'case_id': i,
            'cvr_message': f"Test utterance {i} for {label}",
            'label': label,
            'turn_number': 0,
        })
    
    df = pd.DataFrame(dummy_data)
    
    print("\nOriginal distribution:")
    print(df['label'].value_counts())
    
    # Apply SMOTE
    sampling_strategy = {
        'NORMAL': 1.0,
        'EARLY_WARNING': 0.7,
        'ELEVATED': 0.5,
        'CRITICAL': 0.5,
    }
    
    augmented = apply_smote_to_dataset(df, sampling_strategy=sampling_strategy)
    
    print("\nAugmented distribution:")
    from collections import Counter
    labels = [seq['label'] for seq in augmented]
    print(Counter(labels))
