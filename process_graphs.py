import networkx as nx
from collections import defaultdict
import re
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from typing import List, Dict, Set, Tuple
import json
import os 
import hashlib
import datetime 
import concurrent 
import networkx as nx
from glob import glob
import multiprocessing as mp
import gzip 
import torch 

def convert_node_features_to_tensor(features_dict_list):
    """Convert list of feature dictionaries to a tensor with consistent dimensions."""
    # Define the order of features to ensure consistency
    feature_keys = [
        'mem_ops', 'calls', 'instructions', 'stack_ops', 'reg_writes',
        'external_calls', 'internal_calls', 'mem_reads', 'mem_writes',
        'in_degree', 'out_degree', 'is_conditional', 'has_jump', 'has_ret'
    ]
    
    # Convert each dictionary to a list with consistent order
    features_list = []
    for feat_dict in features_dict_list:
        node_features = [float(feat_dict.get(key, 0)) for key in feature_keys]
        features_list.append(node_features)
    
    # Convert to tensor
    return torch.tensor(features_list, dtype=torch.float)


class BAPInstruction:
    """Parser for BAP IL instructions."""
    
    def __init__(self, instruction_str: str):
        self.raw = instruction_str
        self.address = self._extract_address()
        self.operation = self._parse_operation()
        self.operands = self._parse_operands()
        
    def _extract_address(self) -> str:
        """Extract instruction address."""
        addr_match = re.match(r'([0-9a-fA-F]+):', self.raw)
        return addr_match.group(1) if addr_match else None
        
    def _parse_operation(self) -> dict:
        """Parse the operation type and flags."""
        ops = {
            'mem_read': 'mem[' in self.raw,
            'mem_write': ':= mem with' in self.raw,
            'call': 'call' in self.raw,
            'conditional': any(flag in self.raw for flag in ['CF', 'ZF', 'SF', 'OF']),
            'arithmetic': any(op in self.raw for op in ['+', '-', '*', '/', '<<', '>>']),
            'assignment': ':=' in self.raw,
            'stack': any(reg in self.raw for reg in ['RSP', 'ESP', 'SP'])
        }
        return ops
        
    def _parse_operands(self) -> dict:
        """Parse instruction operands."""
        operands = {
            'registers': self._extract_registers(),
            'constants': self._extract_constants(),
            'addresses': self._extract_addresses(),
            'function_calls': self._extract_function_calls()
        }
        return operands
        
    def _extract_registers(self) -> Set[str]:
        """Extract register names."""
        register_pattern = r'\b(R[A-Z0-9]{2}|E[A-Z]{2}|[A-Z]{2})\b'
        return set(re.findall(register_pattern, self.raw))
        
    def _extract_constants(self) -> Set[str]:
        """Extract constant values."""
        const_pattern = r'#([0-9a-fA-F]+)'
        return set(re.findall(const_pattern, self.raw))
        
    def _extract_addresses(self) -> Set[str]:
        """Extract memory addresses."""
        addr_pattern = r'0x([0-9a-fA-F]+)'
        return set(re.findall(addr_pattern, self.raw))
        
    def _extract_function_calls(self) -> Set[str]:
        """Extract function call targets."""
        if 'call' not in self.raw:
            return set()
        call_pattern = r'call\s+([^\s]+)'
        return set(re.findall(call_pattern, self.raw))

class MalwareFeatureExtractor:
    """Extract malware-specific features from CFGs."""
    
    def __init__(self):
        # Known suspicious patterns
        self.suspicious_apis = {
            'CreateProcess', 'VirtualAlloc', 'WriteProcessMemory',
            'CreateRemoteThread', 'GetProcAddress', 'LoadLibrary'
        }
        self.suspicious_strings = {
            'cmd.exe', 'powershell', 'rundll32', 'temp', '%temp%'
        }
        
    def extract_features(self, G: nx.DiGraph) -> dict:
        """Extract malware-specific features from the graph."""
        features = {
            'api_patterns': self._analyze_api_calls(G),
            'structural_patterns': self._analyze_structural_patterns(G),
            'complexity_metrics': self._compute_complexity_metrics(G),
            'obfuscation_indicators': self._detect_obfuscation(G)
        }
        return features
        
    def _analyze_api_calls(self, G: nx.DiGraph) -> dict:
        """Analyze API call patterns."""
        api_calls = defaultdict(int)
        suspicious_calls = defaultdict(int)
        
        for _, data in G.nodes(data=True):
            for call in data.get('calls', []):
                if isinstance(call, str):
                    api_name = call.split('call')[1].strip().split()[0]
                    api_calls[api_name] += 1
                    if any(sus in api_name for sus in self.suspicious_apis):
                        suspicious_calls[api_name] += 1
                        
        return {
            'total_api_calls': len(api_calls),
            'unique_apis': len(set(api_calls.keys())),
            'suspicious_api_count': sum(suspicious_calls.values()),
            'suspicious_apis_detected': list(suspicious_calls.keys())
        }
        
    def _analyze_structural_patterns(self, G: nx.DiGraph) -> dict:
        """Analyze structural patterns common in malware."""
        return {
            'has_loops': not nx.is_directed_acyclic_graph(G),
            'max_loop_depth': self._compute_loop_depth(G),
            'branch_complexity': self._compute_branch_complexity(G),
            'indirect_calls': self._count_indirect_calls(G)
        }
        
    def _compute_complexity_metrics(self, G: nx.DiGraph) -> dict:
        """Compute various complexity metrics."""
        return {
            'cyclomatic_complexity': nx.number_of_edges(G) - nx.number_of_nodes(G) + 2,
            'avg_degree': sum(dict(G.degree()).values()) / max(1, G.number_of_nodes()),
            'density': nx.density(G),
            'strongly_connected_components': len(list(nx.strongly_connected_components(G)))
        }
        
    def _detect_obfuscation(self, G: nx.DiGraph) -> dict:
        """Detect potential obfuscation techniques."""
        indicators = {
            'dead_code': self._detect_dead_code(G),
            'instruction_overlap': self._detect_instruction_overlap(G),
            'unusual_constants': self._detect_unusual_constants(G)
        }
        return indicators
        
    def _compute_loop_depth(self, G: nx.DiGraph) -> int:
        """Compute maximum loop nesting depth."""
        if nx.is_directed_acyclic_graph(G):
            return 0
        cycles = nx.simple_cycles(G)
        max_depth = 0
        for cycle in cycles:
            contained_cycles = sum(1 for other_cycle in nx.simple_cycles(G) 
                                if set(cycle).issuperset(set(other_cycle)))
            max_depth = max(max_depth, contained_cycles)
        return max_depth
        
    def _compute_branch_complexity(self, G: nx.DiGraph) -> float:
        """Compute branching complexity."""
        out_degrees = [d for _, d in G.out_degree()]
        return np.mean(out_degrees) if out_degrees else 0
        
    def _count_indirect_calls(self, G: nx.DiGraph) -> int:
        """Count indirect function calls."""
        count = 0
        for _, data in G.nodes(data=True):
            for call in data.get('calls', []):
                if 'call mem[' in str(call):
                    count += 1
        return count
        
    def _detect_dead_code(self, G: nx.DiGraph) -> dict:
        """Detect potential dead code blocks."""
        reachable = set(nx.descendants(G, list(G.nodes())[0]))
        unreachable = set(G.nodes()) - reachable
        return {
            'unreachable_blocks': len(unreachable),
            'unreachable_ratio': len(unreachable) / max(1, G.number_of_nodes())
        }
        
    def _detect_instruction_overlap(self, G: nx.DiGraph) -> dict:
        """Detect potential instruction overlapping."""
        overlaps = 0
        for node, data in G.nodes(data=True):
            instructions = data.get('instructions', [])
            addresses = set()
            for instr in instructions:
                if isinstance(instr, str):
                    addr_match = re.match(r'([0-9a-fA-F]+):', instr)
                    if addr_match:
                        addr = int(addr_match.group(1), 16)
                        if addr in addresses:
                            overlaps += 1
                        addresses.add(addr)
        return {'total_overlaps': overlaps}
        
    def _detect_unusual_constants(self, G: nx.DiGraph) -> dict:
        """Detect potentially suspicious constants."""
        constants = []
        for _, data in G.nodes(data=True):
            for instr in data.get('instructions', []):
                if isinstance(instr, str):
                    consts = re.findall(r'#([0-9a-fA-F]+)', instr)
                    constants.extend(int(c, 16) for c in consts)
        
        return {
            'unique_constants': len(set(constants)),
            'high_entropy_constants': sum(1 for c in constants if bin(c).count('1') / len(bin(c)) > 0.4)
        }

class ParallelBAPProcessor:
    def __init__(self, num_workers=None, output_path='cfg_analysis'):
        self.num_workers = num_workers or mp.cpu_count()
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize progress tracking
        self.processed_files = set()
        self._load_progress()

    def _load_progress(self):
        """Load list of already processed files."""
        progress_file = os.path.join(self.output_path, 'processed_files.txt')
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                self.processed_files = set(line.strip() for line in f)

    def _save_progress(self, cfg_file):
        """Save progress after each file is processed."""
        progress_file = os.path.join(self.output_path, 'processed_files.txt')
        with open(progress_file, 'a') as f:
            f.write(f"{cfg_file}\n")
        self.processed_files.add(cfg_file)

    def parse_and_save(self, cfg_file: str) -> bool:
        """Parse a single BAP CFG file and save its graph structure efficiently."""
        try:
            with open(cfg_file, 'r') as f:
                content = f.read()

            G = nx.DiGraph()
                
            # Parse nodes
            node_pattern = r'"\\%(.*?)"(\[label="(.*?)"\])?'
            for match in re.finditer(node_pattern, content):
                node_id, _, label = match.groups()
                if node_id:
                    node_attrs = {
                        'instructions': [],
                        'mem_ops': [],
                        'calls': []
                    }
                    
                    if label:
                        for line in label.split('\\l'):
                            if ':' in line:
                                instr = line.split(':', 1)[1].strip()
                                node_attrs['instructions'].append(instr)
                                if 'mem' in instr:
                                    node_attrs['mem_ops'].append(instr)
                                if 'call' in instr:
                                    node_attrs['calls'].append(instr)
                    
                    G.add_node(node_id, **node_attrs)
            
            # Parse edges
            edge_pattern = r'"\\%(.*?)" -> "\\%(.*?)"(\[label="(.*?)"\])?'
            for match in re.finditer(edge_pattern, content):
                src, dst, _, label = match.groups()
                if src in G.nodes() and dst in G.nodes():
                    G.add_edge(src, dst, condition=label if label else None)

            if not G.number_of_nodes():
                print(f"Warning: Empty graph for {cfg_file}")
                return False

            # Create node mapping and node features
            node_mapping = {node: idx for idx, node in enumerate(G.nodes())}
            
            # Create edge list in COO format and edge features
            edge_index = []
            edge_features = []
            for src, dst, data in G.edges(data=True):
                edge_index.append([node_mapping[src], node_mapping[dst]])
                edge_features.append({'condition': data.get('condition', None)})
            
            # Create node features with consistent feature set
            node_features = []
            for node in G.nodes():
                node_data = G.nodes[node]
                # Initialize all features to 0
                features = {
                    'mem_ops': 0,
                    'calls': 0,
                    'instructions': 0,
                    'stack_ops': 0,
                    'reg_writes': 0,
                    'external_calls': 0,
                    'internal_calls': 0,
                    'mem_reads': 0,
                    'mem_writes': 0,
                    'in_degree': 0,
                    'out_degree': 0,
                    'is_conditional': 0,
                    'has_jump': 0,
                    'has_ret': 0
                }
                
                # Update with actual values
                features.update({
                    'mem_ops': len(node_data.get('mem_ops', [])),
                    'calls': len(node_data.get('calls', [])),
                    'instructions': len(node_data.get('instructions', [])),
                    'stack_ops': sum(1 for instr in node_data.get('instructions', []) 
                                if 'RSP' in instr or 'ESP' in instr),
                    'reg_writes': sum(1 for instr in node_data.get('instructions', []) 
                                    if ':=' in instr),
                    'external_calls': sum(1 for call in node_data.get('calls', []) 
                                        if ':external' in call),
                    'internal_calls': sum(1 for call in node_data.get('calls', []) 
                                        if ':external' not in call),
                    'mem_reads': sum(1 for op in node_data.get('mem_ops', []) 
                                if 'mem[' in op and ':=' not in op),
                    'mem_writes': sum(1 for op in node_data.get('mem_ops', []) 
                                    if 'mem with' in op),
                    'in_degree': G.in_degree(node),
                    'out_degree': G.out_degree(node),
                    'is_conditional': int(any('CF' in instr or 'ZF' in instr 
                                        for instr in node_data.get('instructions', []))),
                    'has_jump': int(any('jmp' in instr.lower() 
                                    for instr in node_data.get('instructions', []))),
                    'has_ret': int(any('ret' in instr.lower() 
                                    for instr in node_data.get('instructions', [])))
                })
                
                node_features.append(features)

            # Save with compression
            print(f"Processing {cfg_file}")
            filename = cfg_file.rsplit('/', 1)[-1]
            hash_part = filename.split('_')[0]
            print(hash_part)

            output_file = os.path.join(self.output_path, f"{hash_part}.json.gz")
            print(f"Saving to {output_file}")
                        
            data = {
                'file': cfg_file,
                'graph_structure': {
                    'num_nodes': G.number_of_nodes(),
                    'node_features': convert_node_features_to_tensor(node_features).tolist(),
                    'edge_index': edge_index,
                    'edge_features': [f for f in edge_features if f['condition'] is not None],
                    'node_mapping': node_mapping
                },
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            with gzip.open(output_file, 'wt') as f:
                json.dump(data, f, separators=(',', ':'))

            return True

        except Exception as e:
            print(f"Error processing {cfg_file}: {str(e)}")
            with open(os.path.join(self.output_path, 'errors.log'), 'a') as f:
                f.write(f"{datetime.datetime.now()}: Error in {cfg_file}: {str(e)}\n")
            return False
            
    def process_batch(self, cfg_patterns):
        """Process files matching one or more patterns in parallel.
        
        Args:
            cfg_patterns: String pattern or list of patterns
        """
        # Handle both single pattern (str) and multiple patterns (list)
        if isinstance(cfg_patterns, str):
            patterns = [cfg_patterns]
        else:
            patterns = cfg_patterns
            
        # Collect all files from all patterns
        cfg_files = []
        for pattern in patterns:
            cfg_files.extend(glob(pattern))
        
        if not cfg_files:
            print(f"No files found matching patterns: {patterns}")
            return
            
        # Filter out already processed files
        remaining_files = [f for f in cfg_files if f not in self.processed_files]
        print(f"Found {len(remaining_files)} unprocessed files out of {len(cfg_files)} total files")

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_file = {
                executor.submit(self.parse_and_save, cfg_file): cfg_file 
                for cfg_file in remaining_files
            }
            
            for future in concurrent.futures.as_completed(future_to_file):
                cfg_file = future_to_file[future]
                try:
                    if future.result():
                        self._save_progress(cfg_file)
                        print(f"Successfully processed: {cfg_file}")
                except Exception as e:
                    print(f"Error in future for {cfg_file}: {e}")

    # def combine_results(self):
    #     """Combine all individual JSON files into a single analysis file."""
    #     all_results = []
    #     for filename in os.listdir(self.output_path):
    #         if filename.endswith('.json') and filename != 'combined_analysis.json':
    #             filepath = os.path.join(self.output_path, filename)
    #             with open(filepath, 'r') as f:
    #                 all_results.append(json.load(f))
                    
    #     with open(os.path.join(self.output_path, 'combined_analysis.json'), 'w') as f:
    #         json.dump(all_results, f, indent=2)


    def combine_results(self):
        """Combine all individual JSON files into a single analysis file."""
        all_results = []
        for filename in os.listdir(self.output_path):
            if filename.endswith('.json.gz') and filename != 'combined_analysis.json.gz':
                filepath = os.path.join(self.output_path, filename)
                with gzip.open(filepath, 'rt') as f:
                    all_results.append(json.load(f))
                        
        with gzip.open(os.path.join(self.output_path, 'combined_analysis.json.gz'), 'wt') as f:
            json.dump(all_results, f, separators=(',', ':'))

def main():
    # Initialize processor
    processor = ParallelBAPProcessor(
        num_workers=mp.cpu_count(),
        output_path='cfg_analysis_results2'
    )
    
    # Define path pattern for CFG files
    cfg_pattern = '/large/bodmas/exe_cfg/*_refang_cfg.exe'
    
    # Process files
    processor.process_batch(cfg_pattern)
    
    # Optionally combine results
    #processor.combine_results()


if __name__ == "__main__":
    main()
