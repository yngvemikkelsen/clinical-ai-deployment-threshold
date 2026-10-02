# v1.2.2 — cross-reference fix in Multimedia Appendix 4

Patch release. One defect, introduced by v1.2.1 and found by comparing the built
appendices against the versions currently held by the journal.

## Fixed

- `build/mma_src/a4_protocol.txt` — two prose cross-references still read
  "Table 4.1" after v1.2.1 restored the submitted "Table S1" numbering to the
  table captions. The built Appendix 4 therefore captioned its table "Table S1"
  and then twice directed the reader to "Table 4.1". Both now read "Table S1",
  matching the caption and the version held by the journal.
- `appendices/MMA4_query_generation.docx` — rebuilt from the corrected source.

All nine appendices were swept for the same fault; no other old-style
"Table N.M" reference remains in any of them.

## Note on provenance

Appendix 4 joins Appendices 2 and 5 as a document that had been hand-finished
after generation: its source has carried "Table 4.1" since the numbering
convention changed, while the committed document said "Table S1", and the
discrepancy stayed invisible because the document was never rebuilt. The source
is now correct, so the two cannot diverge again.

No analysis was re-run. No numerical result changed.
