from __future__ import annotations

import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    conditions: tuple[str, ...]
    conclusion: str
    explanation: str


RULES: list[Rule] = [
    Rule("R1", "Виявлення потреби в очищенні даних",
         ("has_missing_values",), "needs_missing_value_handling",
         "У наборі є пропущені значення, тому перед навчанням потрібно визначити спосіб їх обробки."),
    Rule("R2", "Виявлення потреби в кодуванні категоріальних ознак",
         ("has_categorical_features",), "needs_categorical_encoding",
         "Набір містить категоріальні ознаки, тому потрібно визначити спосіб їх кодування."),
    Rule("R3", "Готовність даних до навчання",
         ("data_cleaned", "features_prepared", "target_available"), "data_ready_for_training",
         "Дані очищені, ознаки підготовлені, а цільова змінна доступна."),
    Rule("R4", "Можливість навчання класифікатора",
         ("data_ready_for_training", "classification_task_defined"), "can_train_classifier",
         "Є підготовлені дані та визначена задача класифікації."),
    Rule("R5", "Потреба у незалежному тестуванні",
         ("model_trained",), "needs_independent_evaluation",
         "Навчену модель необхідно перевірити на даних, які не використовувалися для навчання."),
    Rule("R6", "Можливість побудови матриці помилок",
         ("predictions_available", "true_labels_available"), "can_build_confusion_matrix",
         "Наявні прогнози та фактичні класи, тому можна порівняти їх у матриці помилок."),
    Rule("R7", "Модель не відповідає вимогам якості",
         ("model_evaluated", "quality_below_threshold"), "needs_model_improvement",
         "Оцінювання показало якість нижче встановленого порога, тому модель або дані потрібно переглянути."),
    Rule("R8", "Об'єкт не має визначеного сегмента",
         ("property_features_available", "price_segment_missing"), "needs_segment_prediction",
         "Характеристики об'єкта доступні, але ціновий сегмент відсутній; потрібно виконати класифікацію."),
    Rule("R9", "Потреба в моніторингу",
         ("model_deployed",), "needs_model_monitoring",
         "Після впровадження моделі потрібно контролювати її роботу та якість прогнозів."),
    Rule("R10", "Погіршення моделі після впровадження",
         ("model_deployed", "quality_degraded"), "needs_drift_investigation",
         "Якість моделі погіршилася після впровадження, тому потрібно перевірити зміни в даних і роботі моделі."),
    Rule("R11", "Пропущені значення блокують готовність даних",
         ("has_missing_values", "not_missing_values_handled"), "data_not_ready",
         "Пропущені значення ще не оброблено, тому набір не можна вважати повністю підготовленим."),
    Rule("R12", "Категоріальні ознаки ще не підготовлені",
         ("has_categorical_features", "not_categorical_features_encoded"), "data_not_ready",
         "Категоріальні ознаки ще не закодовані, тому дані потребують додаткової підготовки."),
    # --- нові правила за результатами перевірки якості (розділ 5) ---
    Rule("R13", "Дозвіл на використання моделі",
         ("model_trained", "model_evaluated_independent", "quality_acceptable"), "model_approved_for_use",
         "Модель навчена, перевірена на незалежних даних і має прийнятну якість."),
    Rule("R14", "Можливість прогнозу сегмента",
         ("needs_segment_prediction", "model_approved_for_use", "required_features_complete"),
         "can_predict_segment",
         "Об'єкт без сегмента, модель допущена до використання, а потрібні ознаки повні."),
    Rule("R15", "Зміна розподілу даних",
         ("model_deployed", "data_drift_detected"), "needs_quality_reevaluation",
         "Розподіл даних змінився: це привід повторно оцінити якість, а не доказ погіршення."),
]

# Пари фактів, які не можуть бути істинними одночасно.
INCOMPATIBLE: list[tuple[str, str]] = [
    ("data_ready_for_training", "data_not_ready"),
    ("model_approved_for_use", "needs_model_improvement"),
]

SCENARIOS: list[tuple[str, set[str]]] = [
    ("1. Набір містить пропущені значення",
     {"has_missing_values", "not_missing_values_handled",
      "has_categorical_features", "not_categorical_features_encoded"}),
    ("2. Дані підготовлені до навчання",
     {"data_cleaned", "features_prepared", "target_available", "classification_task_defined"}),
    ("3. Модель навчена та оцінюється",
     {"model_trained", "predictions_available", "true_labels_available"}),
    ("4. Модель має низьку якість",
     {"model_evaluated", "quality_below_threshold"}),
    ("5. Якість моделі погіршилася після впровадження",
     {"model_deployed", "quality_degraded"}),
    ("6. Об'єкт без сегмента, модель допущена до використання",
     {"property_features_available", "price_segment_missing", "model_trained",
      "model_evaluated_independent", "quality_acceptable", "required_features_complete"}),
    ("7. Після впровадження змінився розподіл даних",
     {"model_deployed", "data_drift_detected"}),
    ("8. Модель навчена, але не перевірена на незалежних даних",
     {"model_trained"}),
]


def forward_chain(initial_facts: Iterable[str], rules: list[Rule], max_iterations: int = 20):
    """Пряме виведення. Повертає факти, трасування та джерело кожного виведеного факту."""
    facts = set(initial_facts)
    origin: dict[str, Rule] = {}
    trace: list[dict[str, str]] = []
    fired: set[str] = set()

    for iteration in range(1, max_iterations + 1):
        new: list[Rule] = []
        for rule in rules:
            if rule.rule_id in fired:
                continue
            if all(c in facts for c in rule.conditions):
                fired.add(rule.rule_id)
                if rule.conclusion not in facts:
                    new.append(rule)
        if not new:
            break
        for rule in new:
            facts.add(rule.conclusion)
            origin.setdefault(rule.conclusion, rule)
            trace.append({"iteration": str(iteration), "rule": rule.rule_id,
                          "description": rule.description,
                          "conditions": " AND ".join(rule.conditions),
                          "conclusion": rule.conclusion, "why": rule.explanation})
    return facts, trace, origin


def explain(fact: str, origin: dict[str, Rule], initial: set[str], depth: int = 1) -> None:
    """Пояснення «чому»: ланцюжок правил і фактів, що привели до висновку."""
    pad = "  " * depth
    if fact in initial:
        print(f"{pad}• {fact} — початковий факт")
        return
    rule = origin[fact]
    print(f"{pad}• {fact} ← {rule.rule_id}: {' AND '.join(rule.conditions)}")
    for cond in rule.conditions:
        explain(cond, origin, initial, depth + 1)


def check_incompatible(facts: set[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in INCOMPATIBLE if a in facts and b in facts]


def print_scenario(name: str, initial: set[str]) -> None:
    print("\n" + "=" * 78)
    print(f"СЦЕНАРІЙ: {name}")
    print("- Початкові факти:")
    for f in sorted(initial):
        print(f"  • {f}")
    facts, trace, origin = forward_chain(initial, RULES)

    print("\nТрасування:")
    if not trace:
        print("  Жодне правило не спрацювало.")
    for t in trace:
        print(f"  Ітерація {t['iteration']} | {t['rule']}: {t['description']}\n"
              f"    Умова: {t['conditions']}\n    Висновок: {t['conclusion']}\n"
              f"    Пояснення: {t['why']}")

    derived = sorted(facts - initial)
    print("\nЧому (пояснення виведених фактів):")
    if not derived:
        print("  Нових фактів не виведено.")
    for f in derived:
        explain(f, origin, initial)

    print("\nПідсумкові факти:")
    for f in sorted(facts):
        print(f"  • {f}")
    bad = check_incompatible(facts)
    for a, b in bad:
        print(f"  ! СУПЕРЕЧНІСТЬ: {a} і {b}")


def check_rule_quality(rules: list[Rule]) -> None:
    print("\n" + "=" * 78)
    print("ПЕРЕВІРКА ЯКОСТІ БАЗИ ПРАВИЛ")
    print(f"Кількість правил: {len(rules)}")

    sigs = Counter((tuple(sorted(r.conditions)), r.conclusion) for r in rules)
    dups = [s for s, c in sigs.items() if c > 1]
    if dups:
        print("Дублікати:", dups)
    else:
        print("Повних дублікатів за умовами та висновками не знайдено.")

    groups: dict[tuple[str, ...], list[Rule]] = {}
    for r in rules:
        groups.setdefault(tuple(sorted(r.conditions)), []).append(r)
    conflicts = [g for g in groups.values() if len({r.conclusion for r in g}) > 1]
    if conflicts:
        for g in conflicts:
            print("Конфлікт:", ", ".join(f"{r.rule_id}->{r.conclusion}" for r in g))
    else:
        print("Правил з однаковими умовами та різними висновками не знайдено.")

    # Суперечності на рівні висновків: чи можуть обидва члени пари вивестися разом.
    print("\nПеревірка несумісних висновків (суперечливі вхідні факти):")
    probe = {"data_cleaned", "features_prepared", "target_available",
             "has_missing_values", "not_missing_values_handled"}
    facts, _, _ = forward_chain(probe, rules)
    found = check_incompatible(facts)
    if found:
        for a, b in found:
            print(f"  Знайдено: {a} і {b} виводяться одночасно (R3 проти R11/R12).")
    else:
        print("  Несумісних пар не виведено.")

    used = {c for r in rules for c in r.conditions}
    produced = {r.conclusion for r in rules}
    print("\nПочаткові факти, що не виводяться правилами (мають надходити ззовні):")
    for f in sorted(c for c in used - produced if not c.startswith("not_")):
        print(f"  • {f}")
    unused = sorted(produced - used)
    print("\nКінцеві висновки (не використовуються іншими правилами):")
    for f in unused:
        print(f"  • {f}")


def cohens_kappa(a: list[int], b: list[int]) -> tuple[float, float, float]:
    if len(a) != len(b) or not a:
        raise ValueError("Списки оцінок мають бути однакової ненульової довжини.")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum((ca[c] / n) * (cb[c] / n) for c in set(a) | set(b))
    kappa = 1.0 if p_e == 1 and p_o == 1 else (0.0 if p_e == 1 else (p_o - p_e) / (1 - p_e))
    return kappa, p_o, p_e


def evaluate_expert_agreement(path: Path = Path("C:/Users/Kate/Desktop/Унік/мага/sem_1/magisters_uni/data science/ratings.csv")) -> None:
    print("\n" + "=" * 78)
    print("УЗГОДЖЕНІСТЬ ЕКСПЕРТНИХ ОЦІНОК (раунд 1, незалежні оцінки)")
    if not path.exists():
        print(f"Файл {path} не знайдено. Створіть його за шаблоном (id,statement,expert_a,expert_b).")
        return
    a, b = [], []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            ra, rb = row["expert_a"].strip(), row["expert_b"].strip()
            if ra not in {"0", "1"} or rb not in {"0", "1"}:
                print(f"Оцінки не заповнено або некоректні в рядку {row['id']}. "
                      "Допустимі значення: 0 або 1. Розрахунок зупинено.")
                return
            a.append(int(ra)); b.append(int(rb))
    if len(a) != 20:
        print(f"Очікується 20 оцінок, знайдено {len(a)}. Розрахунок зупинено.")
        return
    kappa, p_o, p_e = cohens_kappa(a, b)
    both1 = sum(x == 1 and y == 1 for x, y in zip(a, b))
    a1b0 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    a0b1 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    both0 = sum(x == 0 and y == 0 for x, y in zip(a, b))
    print("Матриця 2x2 (рядки — експерт A, стовпці — експерт B):")
    print("            B=1   B=0")
    print(f"   A=1     {both1:3d}   {a1b0:3d}")
    print(f"   A=0     {a0b1:3d}   {both0:3d}")
    print(f"Кількість оцінених тверджень: {len(a)}")
    print(f"p_o (спостережений збіг): {p_o:.3f}")
    print(f"p_e (очікуваний збіг):    {p_e:.3f}")
    print(f"Cohen's kappa:            {kappa:.3f}")
    if len(set(a)) == 1 or len(set(b)) == 1:
        pi = (sum(a) + sum(b)) / (2 * len(a))
        ac1 = (p_o - 2 * pi * (1 - pi)) / (1 - 2 * pi * (1 - pi))
        print(f"Додатково (не замінює κ): AC1 Гвета = {ac1:.3f}")


def main() -> None:
    for name, facts in SCENARIOS:
        print_scenario(name, facts)
    check_rule_quality(RULES)
    evaluate_expert_agreement()


if __name__ == "__main__":
    main()