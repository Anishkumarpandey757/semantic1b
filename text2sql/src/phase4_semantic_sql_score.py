"""
Phase 4: Semantic SQL Evaluation

This module provides a semantic_sql_score function to measure the similarity between predicted and ground truth SQL queries.
"""
import re

SQL_KEYWORDS = set("""
SELECT FROM WHERE JOIN INNER LEFT RIGHT FULL OUTER CROSS ON GROUP BY HAVING ORDER LIMIT OFFSET DISTINCT
UNION INTERSECT EXCEPT AS AND OR NOT IN EXISTS LIKE BETWEEN IS NULL COALESCE IFNULL CASE WHEN THEN ELSE END
COUNT SUM AVG MIN MAX
""".split())

CLAUSES = [
    ("SELECT", r"\bSELECT\b"),
    ("FROM", r"\bFROM\b"),
    ("WHERE", r"\bWHERE\b"),
    ("JOIN", r"\bJOIN\b"),
    ("GROUP_BY", r"\bGROUP\s+BY\b"),
    ("HAVING", r"\bHAVING\b"),
    ("ORDER_BY", r"\bORDER\s+BY\b"),
    ("LIMIT", r"\bLIMIT\b"),
    ("OFFSET", r"\bOFFSET\b"),
    ("UNION", r"\bUNION\b"),
    ("INTERSECT", r"\bINTERSECT\b"),
    ("EXCEPT", r"\bEXCEPT\b"),
    ("SUBQUERY", r"\(\s*SELECT\b"),
    ("DISTINCT", r"\bDISTINCT\b"),
    ("AGG", r"\bCOUNT\b|\bSUM\b|\bAVG\b|\bMIN\b|\bMAX\b"),
    ("CASE", r"\bCASE\b"),
]

_table_pat = re.compile(r"\bFROM\s+([`\"\[\]\w\.]+)|\bJOIN\s+([`\"\[\]\w\.]+)", re.IGNORECASE)
_select_pat = re.compile(r"\bSELECT\s+(.*?)\bFROM\b", re.IGNORECASE | re.DOTALL)
_ident_pat = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

def _normalize_sql(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"```sql|```", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r";\s*$", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def _extract_tables(sql: str) -> set:
    sql = _normalize_sql(sql)
    out = set()
    for m in _table_pat.finditer(sql):
        t = (m.group(1) or m.group(2) or "").strip()
        if not t:
            continue
        t = re.sub(r"^[`\"\[]|[`\"\]]$", "", t).split()[0].split(".")[-1]
        out.add(t.lower())
    return out

def _extract_columns(sql: str) -> set:
    sql = _normalize_sql(sql)
    m = _select_pat.search(sql)
    if not m:
        return set()
    sel = re.sub(r"\(.*?\)", " ", m.group(1))  # cheap subselect removal
    ids = {i.lower() for i in _ident_pat.findall(sel)}
    ids = {i for i in ids if i.upper() not in SQL_KEYWORDS}
    ids = {i for i in ids if i not in {"t1","t2","t3","t4","t5","t6","t7","t8","t9","t10"}}
    return ids

def _extract_clauses(sql: str) -> set:
    sql = _normalize_sql(sql)
    out = set()
    for name, pat in CLAUSES:
        if re.search(pat, sql, flags=re.IGNORECASE):
            out.add(name)
    return out

def _extract_keywords(sql: str) -> set:
    sql = _normalize_sql(sql).upper()
    toks = set(re.findall(r"[A-Z_]+", sql))
    return {t for t in toks if t in SQL_KEYWORDS}

def semantic_sql_score(pred: str, gt: str) -> float:
    pred_n, gt_n = _normalize_sql(pred), _normalize_sql(gt)

    tables_p, tables_g = _extract_tables(pred_n), _extract_tables(gt_n)
    cols_p, cols_g     = _extract_columns(pred_n), _extract_columns(gt_n)
    clauses_p, clauses_g = _extract_clauses(pred_n), _extract_clauses(gt_n)
    kw_p, kw_g         = _extract_keywords(pred_n), _extract_keywords(gt_n)

    toks_p = set(re.findall(r"[A-Za-z_]+", pred_n.lower()))
    toks_g = set(re.findall(r"[A-Za-z_]+", gt_n.lower()))
    token_j = _jaccard(toks_p, toks_g)

    score = (
        0.25 * _jaccard(tables_p, tables_g) +
        0.25 * _jaccard(cols_p, cols_g) +
        0.25 * _jaccard(clauses_p, clauses_g) +
        0.15 * _jaccard(kw_p, kw_g) +
        0.10 * token_j
    )
    return score

# Example usage:
# score = semantic_sql_score(pred_sql, ground_truth_sql)
# near_correct = score >= 0.90
