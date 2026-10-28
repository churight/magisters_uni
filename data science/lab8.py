# run_queries.py
import random
from rdflib import Graph
import owlrl

random.seed(42)

PREFIX = """PREFIX ex: <http://example.org/ontologies/real-estate-ml#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

COUNTS = {
 "locationOf":         "SELECT (COUNT(*) AS ?n) WHERE { ?a ex:locationOf ?b }",
 "priceOf":            "SELECT (COUNT(*) AS ?n) WHERE { ?a ex:priceOf ?b }",
 "madeBy":             "SELECT (COUNT(*) AS ?n) WHERE { ?a ex:madeBy ?b }",
 "isLowerThan":        "SELECT (COUNT(*) AS ?n) WHERE { ?a ex:isLowerThan ?b }",
 "ClassifiedObject":   "SELECT (COUNT(*) AS ?n) WHERE { ?a a ex:ClassifiedObject }",
 "HighSegmentObject":  "SELECT (COUNT(*) AS ?n) WHERE { ?a a ex:HighSegmentObject }",
 "EnsembleBasedModel": "SELECT (COUNT(*) AS ?n) WHERE { ?a a ex:EnsembleBasedModel }",
}

def count(g, q):
    return int(next(iter(g.query(PREFIX + q)))[0])

FILE = "C:/Users/Kate/Downloads/real-estate-ml.owl"
raw = Graph().parse(FILE, format="xml")      # без reasoner
inf = Graph().parse(FILE, format="xml")      # з reasoner
owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(inf)

print(f"Тріплів: без = {len(raw)}, з = {len(inf)}\n")
print(f"{'показник':22}{'без':>6}{'з':>6}{'приріст':>9}")
for name, q in COUNTS.items():
    a, b = count(raw, q), count(inf, q)
    print(f"{name:22}{a:>6}{b:>6}{b-a:>9}")

# Приклад: запуск одного запиту (Q3)
q3 = PREFIX + "SELECT ?seg WHERE { ?seg ex:isLowerThan+ ex:HighSegment }"
print("\nQ3 (без reasoner):")
for row in raw.query(q3):
    print(" ", row.seg.split('#')[-1])