---
title: Data Privacy and PHI Handling
source_type: synthetic_policy
version: 1.0
year: 2026
license: CC0-1.0 (synthetic)
---
> SYNTHETIC EDUCATIONAL CONTENT created for demonstration. It paraphrases commonly described principles and is NOT clinical guidance. Replace with your licensed, curated sources.

# Principle
No protected health information (PHI) may enter prompts, logs, traces or the vector index.

# Controls
Documents are screened for identifiers such as names, record numbers, dates of birth, phone numbers and email addresses before indexing. Documents with findings are quarantined for review. User questions are redacted before retrieval and storage. Audit logs store hashes and redacted text, never raw input.

# Vendors
Any external model provider must be covered by a business associate agreement where applicable, and zero-retention options should be used when available.

# Access
Access follows least privilege, and API keys are rotated regularly.
