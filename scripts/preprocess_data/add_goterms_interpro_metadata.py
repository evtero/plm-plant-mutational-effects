#!/usr/bin/env python3
"""
Add GO, InterPro and taxonomy columns to a table of UniProt proteins.

Input:  tab-separated .tsv with an "Id" column of UniProtKB accessions.
Output: same table plus go_*, interpro_*, tax_*, _status (ok, not_found, error) and _notes columns.

Uses UniProt and InterPro APIs.
Taxonomy columns are designed for plants (Viridiplantae).
Rows with _status = error have empty annotations.

Usage: python add_goterms_interpro_metadata.py input.tsv [output.tsv]
"""
import os, sys, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry # re try with backoff for 429, 500, 502, 503, 504

_thread_local = threading.local()
_cache = {}

INTERPRO_TYPES = ["family", "domain", "homologous_superfamily", "conserved_site",
                   "repeat", "active_site", "binding_site", "ptm"]

def get_session():
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=64))
        s.headers.update({"Accept": "application/json"})
        _thread_local.session = s
    return _thread_local.session


def get_json(url):
    try:
        r = get_session().get(url, timeout=15)
    except requests.RequestException:
        return None
    if r.status_code == 404:
        return {}
    if not r.ok or "json" not in r.headers.get("Content-Type", ""):
        return None
    try:
        return r.json()
    except ValueError:
        return None

def cached_interpro(interpro_id):
    if interpro_id in _cache:
        return _cache[interpro_id]
    data = get_json(f"https://www.ebi.ac.uk/interpro/api/entry/interpro/{interpro_id}")
    if data is None:
        return None
    _cache[interpro_id] = data.get("metadata", {})
    return _cache[interpro_id]

def name_of(meta):
    n = meta.get("name")
    return (n.get("short") or n.get("name")) if isinstance(n, dict) else n

def join(lst):
    lst = [str(x) for x in lst if x]
    return "|".join(lst) if lst else None

def get_go_terms(data):
    aspects = {a: {"ids": [], "terms": [], "ev": []} for a in "CPF"}
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "GO":
            continue
        props = {p["key"]: p["value"] for p in ref.get("properties", [])}
        term = props.get("GoTerm", "")
        if ":" not in term:
            continue
        aspect, name = term.split(":", 1)
        if aspect not in aspects:
            continue
        aspects[aspect]["ids"].append(ref.get("id"))
        aspects[aspect]["terms"].append(name.strip())
        aspects[aspect]["ev"].append(props.get("GoEvidenceType", "").split(":")[0])
    return aspects

def get_taxonomy(data):
    tax = data.get("organism", {}) or {}
    lineage = [t.get("scientificName") if isinstance(t, dict) else t
               for t in tax.get("lineage", [])]
    lineage = [n for n in lineage if n]

    tax_class = tax_order = tax_family = tax_clade = None
    for name in lineage:
        if name.endswith(("opsida", "phyceae")):
            tax_class = name
        elif name.endswith("ales"):
            tax_order = name
        elif name.endswith("aceae"):
            tax_family = name

    species = tax.get("scientificName") or ""
    genus = species.split()[0] if species else (lineage[-1] if lineage else None)

    return {
        "tax_class": tax_class, "tax_order": tax_order, "tax_family": tax_family,
        "tax_clade": tax_clade, "tax_genus": genus, "tax_species": species or None,
        "tax_lineage_full": join(lineage),
    }

def get_interpro(data, notes):
    interpro = {t: {"ids": [], "names": []} for t in INTERPRO_TYPES}
    for ref in data.get("uniProtKBCrossReferences", []):
        if ref.get("database") != "InterPro":
            continue
        eid = ref.get("id")
        meta = cached_interpro(eid)
        if meta is None:
            notes.append(f"InterPro:{eid}")
            continue
        itype = (meta.get("type") or "").lower()
        if itype in interpro:
            interpro[itype]["ids"].append(eid)
            interpro[itype]["names"].append(name_of(meta))
    return interpro

def get_protein_info(uid):
    data = get_json(f"https://rest.uniprot.org/uniprotkb/{uid}.json")
    if data is None:
        return {"_status": "error"}
    if not data:
        return {"_status": "not_found"}

    out = {"_status": "ok"}
    notes = []

    try:
        go = get_go_terms(data)
        for a in "CPF":
            out[f"go_{a}_ids"] = join(go[a]["ids"])
            out[f"go_{a}_terms"] = join(go[a]["terms"])
            out[f"go_{a}_evidence"] = join(go[a]["ev"])
    except Exception as e:
        notes.append(f"go:{e}")

    try:
        interpro = get_interpro(data, notes)
        for itype, vals in interpro.items():
            out[f"interpro_{itype}_ids"] = join(vals["ids"])
            out[f"interpro_{itype}_names"] = join(vals["names"])
    except Exception as e:
        notes.append(f"interpro:{e}")

    try:
        out.update(get_taxonomy(data))
    except Exception as e:
        notes.append(f"tax:{e}")

    out["_notes"] = join(notes)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python add_goterms_interpro_metadata.py input.tsv [output.tsv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace(".tsv", ".enriched.tsv")

    df = pd.read_csv(input_file, sep="\t")
    ids = df["Id"].dropna().unique().tolist()
    workers = int(os.getenv("SLURM_CPUS_PER_TASK", "10"))

    results = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(get_protein_info, uid): uid for uid in ids}
        for fut in tqdm(as_completed(futures), total=len(futures)):
            uid = futures[fut]
            try:
                results[uid] = fut.result()
            except Exception as e:
                results[uid] = {"_status": f"error:{e}"}

    enriched = pd.DataFrame.from_dict(results, orient="index")
    enriched.index.name = "Id"
    df_out = df.join(enriched, on="Id", how="left")
    df_out.to_csv(output_file, sep="\t", index=False)
    print(f"[INFO] {output_file}: {len(df_out)} filas, {len(df_out.columns)} columnas")