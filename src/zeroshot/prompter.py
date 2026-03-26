class ZeroShotPrompter:
    """Handles prompt construction for zero-shot classification."""
    
    def __init__(self, class_descriptions: str, labels: list):
        self.class_descriptions = class_descriptions
        self.labels = labels
        self.labels_str = ", ".join(labels)
    
    def build_prompt(self, post_text: str) -> str:
        """
        Build zero-shot classification prompt.
        
        Args:
            post_text: The Reddit post to classify
            
        Returns:
            Complete prompt string
        """
        prompt = f"""
You are a substance use research expert tasked with labeling Reddit posts about opioid use disorder (OUD) into one of the following OUD stages.

Description of classes:
{self.class_descriptions}

### Task
Determine the single OUD stage that describes the author's current situation at the time of writing. Choose exactly one:
{self.labels_str}

### Answer
Return only the label name on one line.

{post_text}
"""
        return prompt
