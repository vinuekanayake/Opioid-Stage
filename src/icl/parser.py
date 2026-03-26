import re
import json
from typing import Optional, Tuple

class ICLOutputParser:
    """Robust parser for GPT-5 outputs in ICL relabeling."""
    
    @staticmethod
    def parse_output(output: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse model output {label:..., rationale:...} into (label, rationale).
        
        Strategy:
        1. Try JSON parsing first
        2. If that fails, fall back to regex-based extraction
        
        Args:
            output: Raw model output string
            
        Returns:
            Tuple of (label, rationale) or (None, None) if parsing fails
        """
        if not output:
            return None, None
        
        # Extract {...} block if present
        m_block = re.search(r"\{.*\}", output, re.DOTALL)
        block = m_block.group(0) if m_block else output.strip()
        
        # --- Step 1: Try JSON parsing ---
        try:
            parsed = json.loads(block)
            label = parsed.get("label")
            rationale = parsed.get("rationale")
            
            # If rationale is a list, join into string
            if isinstance(rationale, list):
                rationale = " ".join(rationale)
            
            return label, rationale
        except Exception:
            pass  # Fall back to regex parsing
        
        # --- Step 2: Regex-based fallback ---
        label = ICLOutputParser._extract_label(block)
        rationale = ICLOutputParser._extract_rationale(block)
        
        return label, rationale
    
    @staticmethod
    def _extract_label(block: str) -> Optional[str]:
        """Extract label using regex."""
        label_regex = re.compile(
            r'["\']?label["\']?\s*:\s*(?:"([^"]+)"|\'([^\']+)\'|([^,}\n]+))',
            re.IGNORECASE
        )
        label_match = label_regex.search(block)
        
        if label_match:
            for g in label_match.groups():
                if g:
                    return g.strip().rstrip(",}")
        return None
    
    @staticmethod
    def _extract_rationale(block: str) -> Optional[str]:
        """Extract rationale using regex."""
        rat_key = re.search(r'["\']?rationale["\']?\s*:', block, re.IGNORECASE)
        
        if not rat_key:
            return None
        
        rest = block[rat_key.end():].lstrip()
        
        # Handle quoted rationale
        if rest.startswith('"') or rest.startswith("'"):
            q = rest[0]
            pos, endpos = 1, None
            
            # Find matching closing quote
            while True:
                nextpos = rest.find(q, pos)
                if nextpos == -1:
                    break
                
                lookahead = rest[nextpos+1: nextpos+6]
                if re.match(r'^\s*(,|\})', lookahead) or (nextpos+1 == len(rest)):
                    endpos = nextpos
                    break
                
                pos = nextpos + 1
            
            if endpos is None:
                endpos = len(rest) - 1
            
            rationale = rest[1:endpos].replace('\\"', '"').replace("\\'", "'").strip()
        else:
            # Handle unquoted rationale
            sep = re.search(r'(,|\})', rest)
            rationale = (rest[:sep.start()] if sep else rest).strip().strip('"').strip("'")
        
        return rationale
    
    @staticmethod
    def format_with_rationale(post_text: str, rationale: str) -> str:
        """
        Format post with rationale marker.
        
        Args:
            post_text: Original post text
            rationale: Extracted rationale
            
        Returns:
            Formatted text with [rationale] marker
        """
        return f"{post_text}[rationale]{rationale}"
