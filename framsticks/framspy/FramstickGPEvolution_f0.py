import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from deap import base, creator, tools, gp
from FramsticksLib import FramsticksLib

FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0
DEFAULT_SIM = "eval-allcriteria.sim;deterministic.sim;sample-period-2.sim;only-body.sim"

Step = Tuple[float, float, float, float]  # dx,dy,dz,rz

Node = List[Step]

def ensure_dir(path: str) -> str:
    if os.path.isdir(path):
        return path
    raise NotADirectoryError(path)

def fmtf(x: float) -> str:
    # stabilne formatowanie dla Framsticks
    if abs(x) < 1e-12:
        x = 0.0
    return f"{x:.3f}".rstrip("0").rstrip(".")

def normalize_f0(geno: str) -> str:
    g = geno.strip()
    # Framsticks rozumie też //0, ale w FramsticksLib bezpiecznie używać /*0*/
    if g.startswith("/*0*/") or g.startswith("//0"):
        return g
    return "/*0*/\n" + g

def genotype_within_constraint(dict_criteria_values, criterion_name, constraint_value):
    if constraint_value is None:
        return True
    actual_value = dict_criteria_values.get(criterion_name, None)
    if actual_value is None:
        return True
    return actual_value <= constraint_value

@dataclass
class Constraints:
    max_numparts: int | None
    max_numjoints: int | None
    max_numneurons: int | None
    max_numconnections: int | None
    max_numgenochars: int | None

def build_f0_chain(steps: List[Step], max_parts: int) -> str:
    # max_parts obejmuje root p: => max_steps = max_parts-1
    max_steps = max(0, max_parts - 1)
    steps = steps[:max_steps]

    lines = ["p:"]  # part 0
    prev = 0
    part = 1
    for dx, dy, dz, rz in steps:
        lines.append("p:")
        # j:p1,p2,dx=...,dy=...,dz=...,rz=...
        fields = [f"{prev},{part}", f"dx={fmtf(dx)}", f"dy={fmtf(dy)}", f"dz={fmtf(dz)}"]
        if abs(rz) > 1e-12:
            fields.append(f"rz={fmtf(rz)}")
        lines.append("j:" + ",".join(fields))
        prev = part
        part += 1

    return normalize_f0("\n".join(lines))

# ---- GP primitives: budowanie listy kroków ----

def _as_steps(x: Node) -> List[Step]:
    # terminale i prymitywy zawsze zwracają listę kroków
    return list(x)

def seq(a: Node, b: Node) -> List[Step]:
    return _as_steps(a) + _as_steps(b)

def rep2(a: Node) -> List[Step]:
    s = _as_steps(a)
    return s + list(s)

def rep3(a: Node) -> List[Step]:
    s = _as_steps(a)
    return s + list(s) + list(s)

# --- terminale jako STAŁE (nie funkcje) ---
EMPTY: List[Step] = []

UP: List[Step]   = [(0.0, 0.0, 1.0, 0.0)]
UP_L: List[Step] = [(0.0, 0.0, 1.0, 0.5)]
UP_R: List[Step] = [(0.0, 0.0, 1.0, -0.5)]
UP_F: List[Step] = [(0.2, 0.0, 0.98, 0.0)]
UP_B: List[Step] = [(-0.2, 0.0, 0.98, 0.0)]

def _eval_compiled(compiled):
    """
    DEAP gp.compile dla arity=0 może zwracać bezpośrednio wynik (np. list),
    a nie funkcję. Ta funkcja ujednolica: zawsze zwraca wynik programu.
    """
    return compiled() if callable(compiled) else compiled

def make_toolbox(frams: FramsticksLib, args) -> base.Toolbox:
    random.seed(args.seed)

    # FitnessMax / Individual dla GP
    try:
        creator.create("FitnessMaxGP", base.Fitness, weights=(1.0,))
    except Exception:
        pass
    try:
        creator.create("IndividualGP", gp.PrimitiveTree, fitness=creator.FitnessMaxGP)
    except Exception:
        pass

    pset = gp.PrimitiveSet("MAIN", arity=0)  # program bez argumentów
    pset.addPrimitive(seq, 2)
    pset.addPrimitive(rep2, 1)
    pset.addPrimitive(rep3, 1)

    # terminale (kroki)
    pset.addTerminal(EMPTY)
    pset.addTerminal(UP)
    pset.addTerminal(UP_L)
    pset.addTerminal(UP_R)
    pset.addTerminal(UP_F)
    pset.addTerminal(UP_B)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genHalfAndHalf, pset=pset, min_=1, max_=args.init_max_depth)
    toolbox.register("individual", tools.initIterate, creator.IndividualGP, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("compile", gp.compile, pset=pset)

    # Operatory GP
    toolbox.register("select", tools.selTournament, tournsize=args.tournament)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=args.mut_max_depth)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # Ograniczenie rozrostu (bloat control)
    toolbox.decorate("mate", gp.staticLimit(key=len, max_value=args.max_nodes))
    toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=args.max_nodes))

    # Ocena
    constraints = Constraints(
        max_numparts=args.max_numparts,
        max_numjoints=args.max_numjoints,
        max_numneurons=args.max_numneurons,
        max_numconnections=args.max_numconnections,
        max_numgenochars=args.max_numgenochars,
    )

    def evaluate(individual):
        try:
            compiled = toolbox.compile(expr=individual)
            steps = _eval_compiled(compiled)
            if not isinstance(steps, list):
                return (FITNESS_VALUE_INFEASIBLE_SOLUTION,)
        except Exception:
            return (FITNESS_VALUE_INFEASIBLE_SOLUTION,)

        geno = build_f0_chain(steps, max_parts=constraints.max_numparts or 30)

        try:
            data = frams.evaluate([geno])[0]
            ev = data["evaluations"][""]
            vertpos = float(ev["vertpos"])
        except Exception:
            return (FITNESS_VALUE_INFEASIBLE_SOLUTION,)

        # ograniczenia (jak w poprzednich labach)
        ev["numgenocharacters"] = len(geno)
        ok = True
        ok &= genotype_within_constraint(ev, "numparts", constraints.max_numparts)
        ok &= genotype_within_constraint(ev, "numjoints", constraints.max_numjoints)
        ok &= genotype_within_constraint(ev, "numneurons", constraints.max_numneurons)
        ok &= genotype_within_constraint(ev, "numconnections", constraints.max_numconnections)
        ok &= genotype_within_constraint(ev, "numgenocharacters", constraints.max_numgenochars)

        if not ok:
            return (FITNESS_VALUE_INFEASIBLE_SOLUTION,)
        return (vertpos,)

    toolbox.register("evaluate", evaluate)
    return toolbox

def save_best_genotype(hof_genotype: str, filename: str, vertpos: float):
    # zapis w formacie *.gen (blokowym) zgodnym z wcześniejszymi analizami
    # Uwaga: f0 jest wielolinijkowe, więc używamy genotype:~ ... ~
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write("org:\n")
        f.write("genotype:~\n")
        f.write(hof_genotype.strip() + "\n")
        f.write("~\n")
        f.write(f"vertpos:{vertpos}\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-path", type=ensure_dir, required=True)
    ap.add_argument("-lib", required=False)
    ap.add_argument("-sim", default=DEFAULT_SIM)
    ap.add_argument("-popsize", type=int, default=50)
    ap.add_argument("-generations", type=int, default=120)
    ap.add_argument("-tournament", type=int, default=7)
    ap.add_argument("-cxpb", type=float, default=0.2)
    ap.add_argument("-mutpb", type=float, default=0.9)
    ap.add_argument("-seed", type=int, default=123)

    # GP kontrola
    ap.add_argument("-init_max_depth", type=int, default=4)
    ap.add_argument("-mut_max_depth", type=int, default=3)
    ap.add_argument("-max_nodes", type=int, default=80)

    # ograniczenia jak wcześniej
    ap.add_argument("-max_numparts", type=int, default=30)
    ap.add_argument("-max_numjoints", type=int, default=60)
    ap.add_argument("-max_numneurons", type=int, default=0)
    ap.add_argument("-max_numconnections", type=int, default=0)
    ap.add_argument("-max_numgenochars", type=int, default=None)

    ap.add_argument("-log_file", required=False)
    ap.add_argument("-hof_savefile", required=False)

    args = ap.parse_args()
    FramsticksLib.DETERMINISTIC = False

    frams = FramsticksLib(args.path, args.lib, args.sim)
    toolbox = make_toolbox(frams, args)

    pop = toolbox.population(n=args.popsize)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("stddev", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    log_data = None
    if args.log_file:
        log_data = {
            "parameters": vars(args),
            "generations": [],
            "best_individual": None,
        }

    # evaluate initial
    invalid = [ind for ind in pop if not ind.fitness.valid]
    for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
        ind.fitness.values = fit
    hof.update(pop)

    feas = [ind for ind in pop if ind.fitness.values[0] != FITNESS_VALUE_INFEASIBLE_SOLUTION]
    if len(feas) == 0:
        rec = {"avg": FITNESS_VALUE_INFEASIBLE_SOLUTION, "stddev": 0.0,
               "min": FITNESS_VALUE_INFEASIBLE_SOLUTION, "max": FITNESS_VALUE_INFEASIBLE_SOLUTION}
    else:
        rec = stats.compile(feas)

    if log_data is not None:
        log_data["generations"].append({"generation": 0, "best_fitness": hof[0].fitness.values[0], **rec})

    start = time.time()
    for gen in range(1, args.generations + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # varAnd (ręcznie)
        for i in range(1, len(offspring), 2):
            if random.random() < args.cxpb:
                offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < args.mutpb:
                offspring[i], = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        hof.update(pop)

        feas = [ind for ind in pop if ind.fitness.values[0] != FITNESS_VALUE_INFEASIBLE_SOLUTION]
        if len(feas) == 0:
            rec = {"avg": FITNESS_VALUE_INFEASIBLE_SOLUTION, "stddev": 0.0,
                   "min": FITNESS_VALUE_INFEASIBLE_SOLUTION, "max": FITNESS_VALUE_INFEASIBLE_SOLUTION}
        else:
            rec = stats.compile(feas)

        if log_data is not None:
            log_data["generations"].append({"generation": gen, "best_fitness": hof[0].fitness.values[0], **rec})

    elapsed = time.time() - start

    # decode best -> f0
    best = hof[0]
    compiled = toolbox.compile(expr=best)
    steps = _eval_compiled(compiled)
    if not isinstance(steps, list):
        steps = []
    geno = build_f0_chain(steps, max_parts=args.max_numparts if args.max_numparts is not None else 30)

    if log_data is not None:
        log_data["elapsed_time"] = elapsed
        log_data["best_individual"] = {"fitness": hof[0].fitness.values[0], "genotype": geno}
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
        with open(args.log_file, "w") as f:
            json.dump(log_data, f, indent=2)

    if args.hof_savefile:
        save_best_genotype(geno, args.hof_savefile, float(hof[0].fitness.values[0]))

    print(f"Best vertpos={hof[0].fitness.values[0]:.6f}")
    print(geno)

if __name__ == "__main__":
    main()