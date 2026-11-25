# Data files

- `fundamentals.csv`: generated from KAP financial report PDFs by `parse_financials.py`.
  `disclosure_date` = date the statement became public on KAP (YYYY-MM-DD), **not** the quarter-end date.
  `revenue` / `ebitda` / `net_profit` = values for that quarter (3 months), `net_debt` = net debt at quarter end,
  `ebitda_ltm` = last-twelve-month EBITDA (for Net Debt/EBITDA). Keep the same unit within a company
  (thousand or million TL). To add a company, append rows in the same format.
- `policy_rate.csv`: CBRT one-week repo rate decisions (`effective_date`, `policy_rate_pct`).
- `fed_rate.csv`: FOMC federal funds target range decisions.
- `cpi.csv`: TurkStat CPI, `publication_date` = release date (template, not loaded yet).

Fields left empty produce a "data not loaded" warning in the tools. Never enter made-up values.
