"""Live ingestion: the four API connectors + the recurring snapshot job.

Credentials come only from the environment (brief Section 11). Connectors
enforce HTTPS, set timeouts, retry transient failures, and never log full URLs
or secrets.
"""
