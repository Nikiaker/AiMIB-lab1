import argparse
import os
import sys
import time
import json
import numpy as np
from deap import creator, base, tools, algorithms
from FramsticksLib import FramsticksLib

# Note: this may be less efficient than running the evolution directly in Framsticks, so if performance is key, compare both options.


FITNESS_VALUE_INFEASIBLE_SOLUTION = -999999.0  # DEAP expects fitness to always be a real value (not None), so this special value indicates that a solution is invalid, incorrect, or infeasible. [Related: https://github.com/DEAP/deap/issues/30 ]. Using float('-inf') or -sys.float_info.max here causes DEAP to silently exit. If you are not using DEAP, set this constant to None, float('nan'), or another special/non-float value to avoid clashing with valid real fitness values, and handle such solutions appropriately as a separate case.


def genotype_within_constraint(genotype, dict_criteria_values, criterion_name, constraint_value):
    REPORT_CONSTRAINT_VIOLATIONS = False
    if constraint_value is not None:
        actual_value = dict_criteria_values[criterion_name]
        if actual_value > constraint_value:
            if REPORT_CONSTRAINT_VIOLATIONS:
                print('Genotype "%s" assigned a special ("infeasible solution") fitness because it violates constraint "%s": %s exceeds the threshold of %s' % (genotype, criterion_name, actual_value, constraint_value))
            return False
    return True


def frams_evaluate(frams_lib, individual):
    FITNESS_CRITERIA_INFEASIBLE_SOLUTION = [FITNESS_VALUE_INFEASIBLE_SOLUTION] * len(OPTIMIZATION_CRITERIA)  # this special fitness value indicates that the solution should not be propagated via selection ("that genotype is invalid"). The floating point value is only used for compatibility with DEAP. If you implement your own optimization algorithm, instead of a negative value in this constant, use a special value like None to properly distinguish between feasible and infeasible solutions.
    genotype = individual[0]  # individual[0] because we can't (?) have a simple str as a DEAP genotype/individual, only list of str.
    data = frams_lib.evaluate([genotype])
    # print("Evaluated '%s'" % genotype, 'evaluation is:', data)
    valid = True
    try:
        first_genotype_data = data[0]
        evaluation_data = first_genotype_data["evaluations"]
        default_evaluation_data = evaluation_data[""]
        fitness = [default_evaluation_data[crit] for crit in OPTIMIZATION_CRITERIA]
    except (KeyError, TypeError) as e:  # the evaluation may have failed for an invalid genotype (such as X[@][@] with "Don't simulate genotypes with warnings" option), or because the creature failed to stabilize, or for some other reason
        valid = False
        print('Problem "%s" so could not evaluate genotype "%s", hence assigned it a special ("infeasible solution") fitness value: %s' % (str(e), genotype, FITNESS_CRITERIA_INFEASIBLE_SOLUTION))
    if valid:
        default_evaluation_data['numgenocharacters'] = len(genotype)  # add one new key to the dictionary for consistent constraint checking below
        valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numparts', parsed_args.max_numparts)
        valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numjoints', parsed_args.max_numjoints)
        valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numneurons', parsed_args.max_numneurons)
        valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numconnections', parsed_args.max_numconnections)
        valid &= genotype_within_constraint(genotype, default_evaluation_data, 'numgenocharacters', parsed_args.max_numgenochars)
    if not valid:
        fitness = FITNESS_CRITERIA_INFEASIBLE_SOLUTION
    return fitness


def frams_crossover(frams_lib, individual1, individual2):
    geno1 = individual1[0]  # individual[0] because we can't (?) have a simple str as a DEAP genotype/individual, only list of str.
    geno2 = individual2[0]  # individual[0] because we can't (?) have a simple str as a DEAP genotype/individual, only list of str.
    individual1[0] = frams_lib.crossOver(geno1, geno2)
    individual2[0] = frams_lib.crossOver(geno1, geno2)
    return individual1, individual2


def frams_mutate(frams_lib, individual):
    individual[0] = frams_lib.mutate([individual[0]])[0]  # individual[0] because we can't (?) have a simple str as a DEAP genotype/individual, only list of str.
    return individual,


def frams_getsimplest(frams_lib, genetic_format, initial_genotype):
    return initial_genotype if initial_genotype is not None else frams_lib.getSimplest(genetic_format)


def is_feasible_fitness_value(fitness_value: float) -> bool:
    assert isinstance(fitness_value, float), f"feasible_fitness({fitness_value}): argument is not of type 'float', it is of type '{type(fitness_value)}'"  # since we are using DEAP, we unfortunately must represent the fitness of an "infeasible solution" as a float...
    return fitness_value != FITNESS_VALUE_INFEASIBLE_SOLUTION  # ...so if a valid solution happens to have fitness equal to this special value, such a solution will be considered infeasible :/


def is_feasible_fitness_criteria(fitness_criteria: tuple) -> bool:
    return all(is_feasible_fitness_value(fitness_value) for fitness_value in fitness_criteria)


def select_feasible(individuals):
    """
    Filters out only feasible individuals (i.e., with fitness different from FITNESS_VALUE_INFEASIBLE_SOLUTION)
    """
    # for ind in individuals:
    #	print(ind.fitness.values, ind)
    feasible_individuals = [ind for ind in individuals if is_feasible_fitness_criteria(ind.fitness.values)]
    count_all = len(individuals)
    count_infeasible = count_all - len(feasible_individuals)
    if count_infeasible != 0:
        print("Selection: ignoring %d infeasible solution%s in a population of size %d" % (count_infeasible, 's' if count_infeasible > 1 else '', count_all))
    return feasible_individuals


def selTournament_only_feasible(individuals, k, tournsize):
    return tools.selTournament(select_feasible(individuals), k, tournsize=tournsize)


def selNSGA2_only_feasible(individuals, k):
    return tools.selNSGA2(select_feasible(individuals), k)  # this method (unfortunately) decreases population size permanently each time an infeasible solution is removed


def prepareToolbox(frams_lib, OPTIMIZATION_CRITERIA, tournament_size, genetic_format, initial_genotype):
    creator.create("FitnessMax", base.Fitness, weights=[1.0] * len(OPTIMIZATION_CRITERIA))
    creator.create("Individual", list, fitness=creator.FitnessMax)  # would be nice to have "str" instead of unnecessary "list of str"

    toolbox = base.Toolbox()
    toolbox.register("attr_simplest_genotype", frams_getsimplest, frams_lib, genetic_format, initial_genotype)  # "Attribute generator"
    # (failed) struggle to have an individual which is a simple str, not a list of str
    # toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_frams)
    # https://stackoverflow.com/questions/51451815/python-deap-library-using-random-words-as-individuals
    # https://github.com/DEAP/deap/issues/339
    # https://gitlab.com/santiagoandre/deap-customize-population-example/-/blob/master/AGbasic.py
    # https://groups.google.com/forum/#!topic/deap-users/22g1kyrpKy8
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_simplest_genotype, 1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", frams_evaluate, frams_lib)
    toolbox.register("mate", frams_crossover, frams_lib)
    toolbox.register("mutate", frams_mutate, frams_lib)
    if len(OPTIMIZATION_CRITERIA) <= 1:
        # toolbox.register("select", tools.selTournament, tournsize=tournament_size) # without explicitly filtering out infeasible solutions - eliminating/discriminating infeasible solutions during selection would only rely on their relatively poor fitness value
        toolbox.register("select", selTournament_only_feasible, tournsize=tournament_size)
    else:
        # toolbox.register("select", selNSGA2) # without explicitly filtering out infeasible solutions - eliminating/discriminating infeasible solutions during selection would only rely on their relatively poor fitness value
        toolbox.register("select", selNSGA2_only_feasible)
    return toolbox


def parseArguments():
    parser = argparse.ArgumentParser(description='Run this program with "python -u %s" if you want to disable buffering of its output.' % sys.argv[0])
    parser.add_argument('-path', type=ensureDir, required=True, help='Path to Framsticks library without trailing slash.')
    parser.add_argument('-lib', required=False, help='Library name. If not given, "frams-objects.dll" (or .so or .dylib) is assumed depending on the platform.')
    parser.add_argument('-sim', required=False, default="eval-allcriteria.sim", help="The name of the .sim file with settings for evaluation, mutation, crossover, and similarity estimation. If not given, \"eval-allcriteria.sim\" is assumed by default. Must be compatible with the \"standard-eval\" expdef. If you want to provide more files, separate them with a semicolon ';'.")

    parser.add_argument('-genformat', required=False, help='Genetic format for the simplest initial genotype, for example 4, 9, or B. If not given, f1 is assumed.')
    parser.add_argument('-initialgenotype', required=False, help='The genotype used to seed the initial population. If given, the -genformat argument is ignored.')

    parser.add_argument('-opt', required=True, help='optimization criteria: vertpos, velocity, distance, vertvel, lifespan, numjoints, numparts, numneurons, numconnections (or other as long as it is provided by the .sim file and its .expdef). For multiple criteria optimization, separate the names by the comma.')
    parser.add_argument('-popsize', type=int, default=50, help="Population size, default: 50.")
    parser.add_argument('-generations', type=int, default=5, help="Number of generations, default: 5.")
    parser.add_argument('-tournament', type=int, default=5, help="Tournament size, default: 5.")
    parser.add_argument('-pmut', type=float, default=0.9, help="Probability of mutation, default: 0.9")
    parser.add_argument('-pxov', type=float, default=0.2, help="Probability of crossover, default: 0.2")
    parser.add_argument('-hof_size', type=int, default=10, help="Number of genotypes in Hall of Fame. Default: 10.")
    parser.add_argument('-hof_savefile', required=False, help='If set, Hall of Fame will be saved in the Framsticks file format (recommended extension *.gen).')

    parser.add_argument('-max_numparts', type=int, default=None, help="Maximum number of Parts. Default: no limit")
    parser.add_argument('-max_numjoints', type=int, default=None, help="Maximum number of Joints. Default: no limit")
    parser.add_argument('-max_numneurons', type=int, default=None, help="Maximum number of Neurons. Default: no limit")
    parser.add_argument('-max_numconnections', type=int, default=None, help="Maximum number of Neural connections. Default: no limit")
    parser.add_argument('-max_numgenochars', type=int, default=None, help="Maximum number of characters in genotype (including the format prefix, if any). Default: no limit")
    
    parser.add_argument('-log_file', required=False, help='JSON file to save evolution log (fitness per generation).')
    parser.add_argument('-stagnation_generations', type=int, default=None, help="Stop evolution if no improvement in Hall of Fame for this many generations. Default: disabled")
    
    return parser.parse_args()


def ensureDir(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)


def save_genotypes(filename, OPTIMIZATION_CRITERIA, hof):
    from framsfiles import writer as framswriter
    with open(filename, "w") as outfile:
        for ind in hof:
            keyval = {}
            for i, k in enumerate(OPTIMIZATION_CRITERIA):  # construct a dictionary with criteria names and their values
                keyval[k] = ind.fitness.values[i]  # TODO it would be better to save in Individual (after evaluation) all fields returned by Framsticks, and get these fields here, not just the criteria that were actually used as fitness in evolution.
            # Note: prior to the release of Framsticks 5.0, saving e.g. numparts (i.e. P) without J,N,C breaks re-calcucation of P,J,N,C in Framsticks and they appear to be zero (nothing serious).
            outfile.write(framswriter.from_collection({"_classname": "org", "genotype": ind[0], **keyval}))
            outfile.write("\n")
    print("Saved '%s' (%d)" % (filename, len(hof)))


def eaSimpleWithLog(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, log_data=None, stagnation_gens=None):
    """
    Modified eaSimple that logs fitness per generation and supports stagnation stopping criterion.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    
    # Helper function to extract fitness value (handles both tuple and scalar)
    def extract_fitness(value):
        if value is None:
            return None
        if isinstance(value, (tuple, list)):
            return value[0] if len(value) > 0 else None
        return value
    
    # Log generation 0
    if log_data is not None:
        best_fit = halloffame[0].fitness.values[0] if halloffame and len(halloffame) > 0 else float('-inf')
        log_data['generations'].append({
            'generation': 0,
            'best_fitness': best_fit,
            'avg_fitness': extract_fitness(record.get('avg')),
            'max_fitness': extract_fitness(record.get('max')),
            'min_fitness': extract_fitness(record.get('min')),
            'std_fitness': extract_fitness(record.get('stddev'))
        })
    
    # Stagnation tracking
    best_fitness_history = []
    if halloffame and len(halloffame) > 0:
        best_fitness_history.append(halloffame[0].fitness.values[0])
    stagnation_counter = 0

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        # Log current generation
        if log_data is not None:
            best_fit = halloffame[0].fitness.values[0] if halloffame and len(halloffame) > 0 else float('-inf')
            log_data['generations'].append({
                'generation': gen,
                'best_fitness': best_fit,
                'avg_fitness': extract_fitness(record.get('avg')),
                'max_fitness': extract_fitness(record.get('max')),
                'min_fitness': extract_fitness(record.get('min')),
                'std_fitness': extract_fitness(record.get('stddev'))
            })
        
        # Check stagnation
        if stagnation_gens is not None and halloffame and len(halloffame) > 0:
            current_best = halloffame[0].fitness.values[0]
            best_fitness_history.append(current_best)
            
            if len(best_fitness_history) > 1 and current_best <= best_fitness_history[-2]:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            
            if stagnation_counter >= stagnation_gens:
                print(f"Evolution stopped due to stagnation: no improvement for {stagnation_gens} generations.")
                if log_data is not None:
                    log_data['stopped_early'] = True
                    log_data['actual_generations'] = gen
                break

    return population, logbook


def main():
    global parsed_args, OPTIMIZATION_CRITERIA  # needed in frams_evaluate(), so made global to avoid passing as arguments

    # random.seed(123)  # see FramsticksLib.DETERMINISTIC below, set to True if you want full determinism
    FramsticksLib.DETERMINISTIC = False  # must be set before the FramsticksLib() constructor call
    parsed_args = parseArguments()
    print("Argument values:", ", ".join(['%s=%s' % (arg, getattr(parsed_args, arg)) for arg in vars(parsed_args)]))
    OPTIMIZATION_CRITERIA = parsed_args.opt.split(",")
    
    # Get mutation rate from frams if available
    mutation_rate = None
    
    framsLib = FramsticksLib(parsed_args.path, parsed_args.lib, parsed_args.sim)
    
    try:
        import frams
        mutation_rate = frams.GenMan.f9_mut._value()
        print(f"Using mutation rate from frams: {mutation_rate}")
    except:
        print("Could not retrieve mutation rate from frams")
    
    toolbox = prepareToolbox(framsLib, OPTIMIZATION_CRITERIA, parsed_args.tournament, '1' if parsed_args.genformat is None else parsed_args.genformat, parsed_args.initialgenotype)
    pop = toolbox.population(n=parsed_args.popsize)
    hof = tools.HallOfFame(parsed_args.hof_size)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    # calculate statistics excluding infeasible solutions (by filtering out these with fitness containing FITNESS_VALUE_INFEASIBLE_SOLUTION)
    filter_feasible_for_function = lambda function, fitness_criteria: function(list(filter(is_feasible_fitness_criteria, fitness_criteria)))
    stats.register("avg", lambda fitness_criteria: filter_feasible_for_function(np.mean, fitness_criteria))
    stats.register("stddev", lambda fitness_criteria: filter_feasible_for_function(np.std, fitness_criteria))
    stats.register("min", lambda fitness_criteria: filter_feasible_for_function(np.min, fitness_criteria))
    stats.register("max", lambda fitness_criteria: filter_feasible_for_function(np.max, fitness_criteria))
    
    # Prepare log data structure
    log_data = None
    if parsed_args.log_file:
        log_data = {
            'parameters': {
                'popsize': parsed_args.popsize,
                'generations': parsed_args.generations,
                'tournament': parsed_args.tournament,
                'pmut': parsed_args.pmut,
                'pxov': parsed_args.pxov,
                'mutation_rate': mutation_rate,
                'optimization_criteria': OPTIMIZATION_CRITERIA,
                'max_numparts': parsed_args.max_numparts,
                'max_numgenochars': parsed_args.max_numgenochars,
                'initialgenotype': parsed_args.initialgenotype
            },
            'generations': [],
            'stopped_early': False,
            'actual_generations': parsed_args.generations
        }
    
    start_time = time.time()
    pop, log = eaSimpleWithLog(pop, toolbox, cxpb=parsed_args.pxov, mutpb=parsed_args.pmut, 
                               ngen=parsed_args.generations, stats=stats, halloffame=hof, 
                               verbose=True, log_data=log_data, 
                               stagnation_gens=parsed_args.stagnation_generations)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    
    print('Best individuals:')
    for ind in hof:
        print(ind.fitness, '\t<--\t', ind[0])
    
    if parsed_args.hof_savefile is not None:
        save_genotypes(parsed_args.hof_savefile, OPTIMIZATION_CRITERIA, hof) # saves a *.gen file, convenient for loading into Framsticks. Otherwise, you can save the HOF in any file format you need.
    
    # Save log file
    if parsed_args.log_file and log_data:
        log_data['elapsed_time'] = elapsed_time
        log_data['best_individual'] = {
            'fitness': hof[0].fitness.values[0] if len(hof) > 0 else None,
            'genotype': hof[0][0] if len(hof) > 0 else None
        }
        
        with open(parsed_args.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        print(f"Log saved to {parsed_args.log_file}")
        print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()