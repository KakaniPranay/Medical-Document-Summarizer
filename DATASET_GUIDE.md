# Building The BiLSTM Training Dataset

This project now includes a small synthetic starter dataset in `data/bilstm_bootstrap_dataset.jsonl`.

Important:
- These examples are synthetic, not real patient records.
- They are useful for bootstrapping the pipeline and understanding the format.
- They are not enough to produce a high-quality clinical summarizer.

## What You Need

For a strong first model, aim for:
- 100 examples minimum
- 300+ examples preferred
- one consistent report style at first, such as discharge summaries or lab summaries

Each example should have:
- `report_id`
- `report_type`
- `text`
- `summary`
- optional `oracle_sentence_ids`
- optional `review_status`

## Best Workflow

1. Start with one report type.
   Good starting choices:
   - `discharge_summary`
   - `lab_summary`
   - `radiology_report`
   - `opd_note`

2. De-identify every report before adding it.
   Remove:
   - patient name
   - phone number
   - address
   - date of birth
   - hospital ID or MRN
   - doctor personal identifiers if not needed

3. Write the summary in a fixed style.
   Keep it:
   - 2 to 4 sentences
   - fact-only
   - faithful to the source
   - no new medical claims
   - focused on diagnosis, findings, treatment, and follow-up

4. Add `oracle_sentence_ids` if possible.
   These are the sentence indices from the original report that best support the summary.
   Example:
   - sentence `0` = first sentence
   - sentence `1` = second sentence

5. Mark review status.
   Suggested values:
   - `draft`
   - `reviewed`
   - `approved`

## Summary Writing Rules

Good summary:
- "The report shows poorly controlled diabetes with elevated HbA1c. Kidney function was normal, and diet plus medication review were advised."

Bad summary:
- "The patient may be developing severe diabetic kidney damage and needs urgent admission."

Why bad:
- it adds facts not present in the report
- it overstates severity

## How To Expand From 20 To 100+

Use this sequence:

1. Review the synthetic examples in `data/bilstm_bootstrap_dataset.jsonl`
2. Copy the format from `data/annotation_template.jsonl`
3. Add 20 of your own de-identified reports
4. Keep only one report type at first
5. Review the summaries for consistency
6. Grow to 50 examples
7. Train and inspect results
8. Expand to 100-300 examples

## Recommended Split

Once you have enough data:
- 70 percent train
- 15 percent validation
- 15 percent test

## Training Command

```bash
python3 train_bilstm.py \
  --input data/bilstm_bootstrap_dataset.jsonl \
  --output-model models/bilstm_extractive.pt \
  --report-dir reports/bilstm_training
```

Replace the input file with your own curated dataset when ready.

## What To Expect

After training:
- `models/bilstm_extractive.pt` will be created
- `reports/bilstm_training/training_report.txt` will show metrics
- the web app will use the trained checkpoint in `bilstm` mode automatically

## Practical Advice

If accuracy matters more than style:
- prioritize better labels over more labels
- keep summaries short and faithful
- avoid mixed formatting styles in the same dataset
- have one reviewer check each summary if possible
