import re
from typing import List, Tuple

class ICLPrompter:
    """Handles prompt construction for ICL relabeling."""
    
    def __init__(self, guidelines: str):
        self.guidelines = guidelines
    
    def extract_post_and_rationale(self, text: str) -> Tuple[str, str]:
        """
        Extracts post without rationale and rationale span separately.
        
        Args:
            text: Full text potentially containing [rationale] marker
            
        Returns:
            Tuple of (post_text, rationale)
        """
        parts = re.split(r"<<LATEX_0>>", text, maxsplit=1)
        if len(parts) == 2:
            post_text, rationale = parts
        else:
            post_text, rationale = text, ""
        return post_text.strip(), rationale.strip()
    
    def format_icl_example(self, post_text: str, label: str, rationale: str) -> str:
        """
        Format a single ICL example.
        
        Args:
            post_text: The post content
            label: The label
            rationale: The rationale (verbatim span)
            
        Returns:
            Formatted example string
        """
        return f'{post_text}\n→\n{{label: {label}, rationale: "{rationale}"}}'
    
    def build_prompt(self, icl_examples: List[str], post_text: str) -> str:
        """
        Build the complete prompt with guidelines, examples, and target post.
        
        Args:
            icl_examples: List of formatted ICL examples
            post_text: The post to label
            
        Returns:
            Complete prompt string
        """
        examples_str = "\n\n".join(icl_examples)
        
        prompt = f"""
You are a substance use research expert tasked with labeling Reddit posts about opioid use disorder (OUD).

{self.guidelines}

### Few-shot Examples

{examples_str}

### Task

Now, label the following post using the same format and rules:

{post_text}
"""
        return prompt
