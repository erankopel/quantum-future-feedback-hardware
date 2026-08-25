#!/usr/bin/env python3
"""Series release gate: refuse to tag unless the deposit metadata is coherent.
Zero dependencies beyond the standard library. Exits non-zero on any failure."""
import json, os, re, sys, hashlib

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fail = []


def chk(cond, msg):
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    if not cond:
        fail.append(msg)


print("RELEASE METADATA GATE")
z = json.load(open(os.path.join(HERE, ".zenodo.json")))
c = open(os.path.join(HERE, "CITATION.cff")).read()

chk(z.get("license") == "MIT", "01 zenodo license is MIT")
chk(z["creators"][0].get("orcid") == "0000-0003-4657-8636", "02 ORCID present and correct")
chk(z["creators"][0].get("affiliation") == "Tel Aviv University", "03 affiliation present")
m = re.search(r'^version:\s*"?([0-9.]+)"?', c, re.M)
chk(bool(m) and m.group(1) == z.get("version"), "04 CITATION.cff and .zenodo.json versions agree")
chk("0000-0003-4657-8636" in c, "05 ORCID in CITATION.cff")
chk(os.path.isfile(os.path.join(HERE, "LICENSE")), "06 LICENSE present")
chk(os.path.isfile(os.path.join(HERE, "requirements.txt")), "07 requirements.txt present")

rid = z.get("related_identifiers", [])
unfilled = [r for r in rid if "XXXX" in str(r.get("identifier", ""))
            or "TO BE FILLED" in str(r.get("identifier", ""))]
chk(not unfilled, "08 no placeholder related_identifiers")
# Zenodo cannot pre-reserve a DOI when the GitHub integration mints it, so the
# first release ships without one and later releases carry it. The gate
# therefore enforces AGREEMENT, not presence: a DOI in one file and a
# different one (or none) in the other is the failure worth catching.
readme = open(os.path.join(HERE, "README.md")).read()
m2 = re.search(r'^doi:\s*"?(10\.\d{4,}/[^"\s]+)"?', c, re.M)
m3 = re.search(r'zenodo\.org/badge/DOI/(10\.\d{4,}/[^)\s]+)\.svg', readme)
cff_doi = m2.group(1) if m2 else None
badge_doi = m3.group(1) if m3 else None
if cff_doi is None and badge_doi is None:
    chk(True, "08b no DOI yet (pre-first-release; Zenodo mints it on release)")
else:
    chk(cff_doi is not None and badge_doi is not None and cff_doi == badge_doi,
        "08b CITATION.cff DOI and README badge agree (cff=%s badge=%s)"
        % (cff_doi, badge_doi))
# Shape test, deliberately holding no identifier: any cloud resource name
# still carrying a 32-hex account or a UUID instance is a leak. Storing the
# real ids here would itself be the thing this check forbids.
CRN = re.compile(r"crn:v1:[^\s\",]*")
HEX32 = re.compile(r"\b[0-9a-f]{32}\b")
UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
leak = []
for root, dirs, files in os.walk(HERE):
    dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
    for f in files:
        if f == "CHECKSUMS.sha256":
            continue
        try:
            body = open(os.path.join(root, f), encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        for crn in CRN.findall(body):
            if HEX32.search(crn) or UUID.search(crn):
                leak.append("%s: %s" % (f, crn[:70]))
chk(not leak, "10 no cloud account or instance identifier in any tracked file")

# checksums cover every tracked file
cs = os.path.join(HERE, "CHECKSUMS.sha256")
if os.path.isfile(cs):
    listed = {l.split(None, 1)[1].strip() for l in open(cs) if l.strip()}
    actual = set()
    for root, dirs, files in os.walk(HERE):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for f in files:
            if f == "CHECKSUMS.sha256":
                continue
            actual.add("./" + os.path.relpath(os.path.join(root, f), HERE))
    chk(listed == actual, "09 CHECKSUMS.sha256 covers exactly the tracked files "
        "(%d listed, %d present)" % (len(listed), len(actual)))
else:
    chk(False, "09 CHECKSUMS.sha256 present")

print("\nRELEASE GATE:", "PASSED" if not fail else "FAILED (%d)" % len(fail))
sys.exit(0 if not fail else 1)
