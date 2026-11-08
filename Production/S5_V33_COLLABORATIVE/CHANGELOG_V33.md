# QAHS V3.3 Change Log

## V3.3 — Collaborative QAHS

- Added `collaborative_qahs_s5.py`.
- Added `collaborative` benchmark condition without modifying V3.2.
- Added evidence-based arbitration between an LLM hypothesis and the QAHS posterior-MAP hypothesis.
- Added a third arbitration outcome: diagnostic experiment selection when the two hypotheses are insufficiently separated by evidence.
- Added bounded LLM self-confidence contribution; confidence cannot override posterior/predictive evidence.
- Added paper and extended launchers for collaborative QAHS.
- Added four-condition comparison support.
- Added arbitration telemetry and hidden-world firewall tests.
- V3.2 remains the frozen baseline.
