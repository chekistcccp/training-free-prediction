REGION_PROPOSAL = """This is one axial contrast-enhanced abdominal/pelvic CT slice with a {grid}x{grid} labeled grid overlaid. The task is NOT to make a definitive diagnosis. Identify up to {top_k} grid cells that contain the strongest visible evidence of a possible primary colorectal tumor abnormality (focal/asymmetric bowel-wall thickening, irregular bowel mass, luminal narrowing, or direct local extension). If no convincing focal colorectal abnormality is visible, return an empty candidate list. Do not use findings outside the colorectum as the target.

Return compact JSON only:
{{
  \"status\": \"NORMAL\" or \"ABNORMAL\" or \"UNCERTAIN\",
  \"candidate_cells\": [\"A1\", \"B2\"],
  \"finding\": \"visible CT evidence only\",
  \"anatomical_region\": \"brief location\"
}}
"""
