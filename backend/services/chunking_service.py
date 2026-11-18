"""
Chunking Service for SERC Security Knowledge Dataset.
Implements RecursiveCharacterTextSplitter and structured domain-aware chunking
across Incident Response Guidance, MITRE ATT&CK, and NIST Cybersecurity Framework 2.0.
Preserves original metadata, source provenance, MITRE technique IDs, and NIST mappings.
"""
import re
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger("serc.chunking_service")


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text by preferred separators (paragraphs, lines, sentences, words)
    to keep chunks contextually coherent while respecting chunk_size and chunk_overlap.
    """

    def __init__(
        self,
        chunk_size: int = 650,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Splits a single text string into chunks."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        return self._split(text, self.separators)

    def _split(self, text: str, separators: List[str]) -> List[str]:
        final_chunks: List[str] = []

        # Find the best separator that exists in the text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split text by chosen separator
        splits = text.split(separator) if separator != "" else list(text)

        # Merge splits up to chunk_size with chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if current_chunk else 0)

            if s_len > self.chunk_size:
                # Sub-split long piece if further separators exist
                if new_separators:
                    sub_pieces = self._split(s, new_separators)
                    for sp in sub_pieces:
                        if current_len + len(sp) + len(separator) <= self.chunk_size:
                            current_chunk.append(sp)
                            current_len += len(sp) + len(separator)
                        else:
                            if current_chunk:
                                final_chunks.append(separator.join(current_chunk).strip())
                                # Overlap logic
                                current_chunk = self._get_overlap_splits(current_chunk, separator)
                                current_len = sum(len(c) + len(separator) for c in current_chunk)
                            current_chunk.append(sp)
                            current_len += len(sp)
                else:
                    # Character slice fallback
                    if current_chunk:
                        final_chunks.append(separator.join(current_chunk).strip())
                        current_chunk = []
                        current_len = 0
                    for start_idx in range(0, len(s), self.chunk_size - self.chunk_overlap):
                        final_chunks.append(s[start_idx:start_idx + self.chunk_size].strip())
                continue

            if current_len + s_len <= self.chunk_size:
                current_chunk.append(s)
                current_len += s_len
            else:
                if current_chunk:
                    final_chunks.append(separator.join(current_chunk).strip())
                    current_chunk = self._get_overlap_splits(current_chunk, separator)
                    current_len = sum(len(c) + len(separator) for c in current_chunk)
                current_chunk.append(s)
                current_len += s_len

        if current_chunk:
            final_chunks.append(separator.join(current_chunk).strip())

        # Clean empty chunks
        return [c for c in final_chunks if c]

    def _get_overlap_splits(self, current_chunk: List[str], separator: str) -> List[str]:
        """Keeps trailing elements that fit within chunk_overlap."""
        overlap_pieces = []
        overlap_len = 0
        for piece in reversed(current_chunk):
            p_len = len(piece) + (len(separator) if overlap_pieces else 0)
            if overlap_len + p_len <= self.chunk_overlap:
                overlap_pieces.insert(0, piece)
                overlap_len += p_len
            else:
                break
        return overlap_pieces


class SecurityKnowledgeChunker:
    """
    Domain-aware chunker that splits security records across 3 core knowledge dimensions:
    1. Incident Response Guidance & Playbook Actions
    2. MITRE ATT&CK Tactics, Techniques, Detection, and Mitigations
    3. NIST Cybersecurity Framework (CSF 2.0) Functions, Categories, and Subcategories

    Preserves parent metadata, source provenance, MITRE technique IDs, and NIST references.
    """

    def __init__(self, chunk_size: int = 650, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )

    def chunk_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Transforms a single cybersecurity dataset row into a list of contextual chunks.
        Each chunk is self-contained, labeled with its knowledge source, and preserves
        complete parent metadata.
        """
        inc_id = str(record.get("incident_id", "")).strip() or "UNKNOWN_INC"
        attack_type = record.get("attack_type", "")
        chunks: List[Dict[str, Any]] = []

        # --- Section 1: Incident Response Guidance ---
        ir_text = self._build_ir_section(record)
        if ir_text:
            ir_pieces = self.text_splitter.split_text(ir_text)
            for idx, piece in enumerate(ir_pieces):
                chunks.append({
                    "knowledge_source": "Incident Response Guidance",
                    "section": "incident_response",
                    "text": piece,
                    "search_text": f"[Incident {inc_id} | Attack: {attack_type} | Incident Response Guidance] {piece}"
                })

        # --- Section 2: MITRE ATT&CK ---
        mitre_text = self._build_mitre_section(record)
        if mitre_text:
            mitre_pieces = self.text_splitter.split_text(mitre_text)
            tid = record.get("mitre_technique_id", "")
            tname = record.get("mitre_technique_name", "")
            for idx, piece in enumerate(mitre_pieces):
                chunks.append({
                    "knowledge_source": "MITRE ATT&CK",
                    "section": "mitre_attack",
                    "text": piece,
                    "search_text": f"[Incident {inc_id} | MITRE {tid} {tname} | Adversary Technique] {piece}"
                })

        # --- Section 3: NIST Cybersecurity Framework 2.0 ---
        nist_text = self._build_nist_section(record)
        if nist_text:
            nist_pieces = self.text_splitter.split_text(nist_text)
            nfunc = record.get("nist_function", "")
            ncat = record.get("nist_category", "")
            for idx, piece in enumerate(nist_pieces):
                chunks.append({
                    "knowledge_source": "NIST CSF 2.0",
                    "section": "nist_csf",
                    "text": piece,
                    "search_text": f"[Incident {inc_id} | NIST CSF 2.0: {nfunc} {ncat}] {piece}"
                })

        # Fallback: if record had none of the above structured fields, chunk combined_knowledge_text
        if not chunks:
            combined = record.get("combined_knowledge_text", "")
            if combined and len(combined.strip()) > 20:
                comb_pieces = self.text_splitter.split_text(combined)
                for piece in comb_pieces:
                    chunks.append({
                        "knowledge_source": record.get("knowledge_set", "Cybersecurity Knowledge"),
                        "section": "combined_knowledge",
                        "text": piece,
                        "search_text": f"[Incident {inc_id} | {attack_type}] {piece}"
                    })

        # Format and attach metadata to every generated chunk
        total_chunks = len(chunks)
        final_chunks: List[Dict[str, Any]] = []

        for idx, ch in enumerate(chunks):
            chunk_id = f"{inc_id}#chunk-{idx}"
            chunk_payload = {
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "total_chunks": total_chunks,
                "incident_id": inc_id,
                "knowledge_source": ch["knowledge_source"],
                "section": ch["section"],
                "attack_type": attack_type,
                "incident_description": record.get("incident_description", ""),
                "target_system": record.get("target_system", ""),
                "outcome": record.get("outcome", ""),
                "attack_severity": record.get("attack_severity", ""),
                "knowledge_set": record.get("knowledge_set", ""),
                "source_type": record.get("source_type", ""),
                "source_reference": record.get("source_reference", ""),
                "mapping_method": record.get("mapping_method", ""),
                "mapping_confidence": record.get("mapping_confidence", ""),
                "mitre_tactic": record.get("mitre_tactic", ""),
                "mitre_technique_id": record.get("mitre_technique_id", ""),
                "mitre_technique_name": record.get("mitre_technique_name", ""),
                "mitre_description": record.get("mitre_description", ""),
                "mitre_mitigations": record.get("mitre_mitigations", ""),
                "mitre_detection": record.get("mitre_detection", ""),
                "nist_function": record.get("nist_function", ""),
                "nist_category": record.get("nist_category", ""),
                "nist_subcategory": record.get("nist_subcategory", ""),
                "nist_description": record.get("nist_description", ""),
                "detection_indicators": record.get("detection_indicators", ""),
                "investigation_steps": record.get("investigation_steps", ""),
                "containment_steps": record.get("containment_steps", ""),
                "eradication_steps": record.get("eradication_steps", ""),
                "recovery_steps": record.get("recovery_steps", ""),
                "recommended_response": record.get("recommended_response", ""),
                "chunk_text": ch["text"],
                "search_text": ch["search_text"]
            }
            final_chunks.append(chunk_payload)

        return final_chunks

    def _build_ir_section(self, record: Dict[str, Any]) -> str:
        parts = []
        if record.get("incident_description"):
            parts.append(f"Incident Context: {record['incident_description']}")
        if record.get("detection_indicators"):
            parts.append(f"Detection Indicators: {record['detection_indicators']}")
        if record.get("investigation_steps"):
            parts.append(f"Investigation Steps: {record['investigation_steps']}")
        if record.get("analysis_steps"):
            parts.append(f"Analysis Steps: {record['analysis_steps']}")
        if record.get("containment_steps"):
            parts.append(f"Containment Steps: {record['containment_steps']}")
        if record.get("eradication_steps"):
            parts.append(f"Eradication Steps: {record['eradication_steps']}")
        if record.get("recovery_steps"):
            parts.append(f"Recovery Steps: {record['recovery_steps']}")
        if record.get("recommended_response"):
            parts.append(f"Recommended Response: {record['recommended_response']}")
        if record.get("lessons_learned"):
            parts.append(f"Lessons Learned: {record['lessons_learned']}")
        return "\n\n".join(parts)

    def _build_mitre_section(self, record: Dict[str, Any]) -> str:
        tid = record.get("mitre_technique_id", "")
        tname = record.get("mitre_technique_name", "")
        tactic = record.get("mitre_tactic", "")
        desc = record.get("mitre_description", "")
        detection = record.get("mitre_detection", "")
        mitigations = record.get("mitre_mitigations", "")

        if not any([tid, tname, desc, detection, mitigations]):
            return ""

        parts = []
        if tid or tname:
            parts.append(f"MITRE ATT&CK Technique: {tid} - {tname} (Tactic: {tactic})")
        if desc:
            parts.append(f"Adversary Description: {desc}")
        if detection:
            parts.append(f"Detection Procedures: {detection}")
        if mitigations:
            parts.append(f"Mitigations: {mitigations}")
        return "\n\n".join(parts)

    def _build_nist_section(self, record: Dict[str, Any]) -> str:
        func = record.get("nist_function", "")
        cat = record.get("nist_category", "")
        subcat = record.get("nist_subcategory", "")
        desc = record.get("nist_description", "")
        examples = record.get("nist_implementation_examples", "")
        refs = record.get("nist_informative_references", "")

        if not any([func, cat, subcat, desc, examples]):
            return ""

        parts = []
        if func or cat:
            parts.append(f"NIST CSF 2.0: Function={func} | Category={cat} | Subcategory={subcat}")
        if desc:
            parts.append(f"NIST Requirement: {desc}")
        if examples:
            parts.append(f"Implementation Examples:\n{examples}")
        if refs:
            parts.append(f"Informative References: {refs}")
        return "\n\n".join(parts)


chunking_service = SecurityKnowledgeChunker()
