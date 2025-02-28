import torch
import time
import numpy as np
import copy
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from indiv_srkit import Individual
from srkit.srkit.primitives import primitives, Primitive
from srkit.srkit.expressions_eqn import Expression
from selection import survivor_selection


def evolve(mconf, pconf, data, device, generations=100, num_parents=100,
           num_children=100, mutation_rate=0.5, cross_rate=0.25, batch_size=64,
           block_size=200,num_workers=1, uniqueID=0):
    gens = generations
    mut_rate = mutation_rate
    cross_rate = cross_rate
    criterion = nn.MSELoss()
    sym_fitness_over_time = np.zeros(gens)
    num_fitness_over_time = np.zeros(gens)
    solution_sym_fitness = torch.inf
    solution_num_fitness = torch.inf
    gen_at_multi = 9999
    idcount = 0


    # print("generating parents", flush=True)
    parents = []
    while len(parents) < num_parents:
        parents.append(Individual(mconf, pconf, device))
    # print("start parent evaluation", flush=True)
    base_model = np.random.choice(25, num_parents, replace=False)
    for parent in parents:
        sym = []
        num = []
        parent.uniqueID = idcount
        parent.encoder.load_state_dict(torch.load('pretrain/Pretrained/encoder'+str(base_model[idcount])+'_best'))
        idcount += 1
        with torch.no_grad():
            loader = DataLoader(data, shuffle=True, pin_memory=True,
                                batch_size=batch_size)

            for it, (p, e0, e1) in enumerate(loader):
                p = p.to(device)  # points
                e0 = e0.to(device)  # input
                e1 = e1.to(device)  # target

                logits, loss = parent.encoder(e0, e1, p, tokenizer=data.itos)
                mean_loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                sym.append(mean_loss.item())

                #get output equations to calculate numeric loss
                errors = []
                # print("starting num loss calc")
                for z in range(logits.size()[0]):
                    eqn = torch.argmax(logits[z], dim=1)
                    eqn_list = eqn.tolist()
                    particular_value = data.endID
                    result = []
                    temp_list = []
                    for op in eqn_list:
                        if op == particular_value:
                            temp_list.append(op)
                            result.append(temp_list)
                            temp_list = []
                        else:
                            temp_list.append(op)
                    result.append(temp_list)
                    eqn_list = result[0]
                    eqn_list = [m for l, m in enumerate(eqn_list) if m != data.startID]
                    clean_eq_list = [m for l, m in enumerate(eqn_list) if m != data.startID]
                    expression = Expression([data.itos[n] for n in clean_eq_list])

                    try:
                        yhats, eqn_pred = expression.execute(p[z][0])
                        errors.append(criterion(yhats, p[z][1]).cpu().numpy())
                        # print(criterion(yhats, p[z][1]))
                        # print(yhats)
                        # print("Valid Equation")
                    except:
                        # print("Invalid Eqn")
                        errors.append(np.inf)
                    for k, err in enumerate(errors):
                        if np.isnan(err):
                            errors[k] = np.inf

                # print(errors)
                num.append(np.mean(errors))
        parent.num_fitness = np.mean(num)
        parent.sym_fitness = np.mean(sym)

    # print("entering evolution", flush=True)
    for gen in range(gens):
        # print("generating children", flush=True)
        children = []
        n = 0
        while len(children) < num_children:
            children.append(copy.deepcopy(parents[n]))
            n += 1

        start = time.time()
        #mutate weights in layers by mutation chance
        for child in children:
            child.uniqueID = idcount
            idcount += 1
            sd = child.sd
            for key, value in sd.items():
                parent2 = np.random.choice(parents)
                while child.uniqueID == parent2.uniqueID:
                    parent2 = np.random.choice(parents)

                #crossover 2 parents
                if 'weight' in key and 'ln' not in key and torch.rand(1)<cross_rate:
                    # print(sd[key])
                    cross_size = int(value.shape[0]/2)
                    # print(value.shape)
                    # print(cross_size)
                    cross_sd = parent2.sd[key][:cross_size]
                    sd[key][:cross_size] = cross_sd
                    # print(sd[key])

                # mutate weights in layers by mutation chance
                if 'weight' in key and 'ln' not in key and torch.rand(1)<mut_rate:
                    mutations = 0.02 * torch.rand(value.size()) - 0.01
                    mut_value = value + mutations.to(device)
                    sd[key] = mut_value
            child.encoder.load_state_dict(sd)

        end = time.time()
        # print("mutation & cross loop", end - start, flush=True)
        #
        start = time.time()
        #evaluate models
        for child in children:
            sym = []
            num = []
            with torch.no_grad():
                loader = DataLoader(data, shuffle=True, pin_memory=True,
                                    batch_size=batch_size)

                for it, (p, e0, e1) in enumerate(loader):
                    p = p.to(device)  # points
                    e0 = e0.to(device)  # input
                    e1 = e1.to(device)  # target

                    logits, loss = parent.encoder(e0, e1, p, tokenizer=data.itos)
                    mean_loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    sym.append(mean_loss.item())

                    # get output equations to calculate numeric loss
                    errors = []
                    # print("starting num loss calc")
                    for z in range(logits.size()[0]):
                        eqn = torch.argmax(logits[z], dim=1)
                        eqn_list = eqn.tolist()
                        particular_value = data.endID
                        result = []
                        temp_list = []
                        for op in eqn_list:
                            if op == particular_value:
                                temp_list.append(op)
                                result.append(temp_list)
                                temp_list = []
                            else:
                                temp_list.append(op)
                        result.append(temp_list)
                        eqn_list = result[0]
                        eqn_list = [m for l, m in enumerate(eqn_list) if m != data.startID]
                        clean_eq_list = [m for l, m in enumerate(eqn_list) if m != data.startID]
                        expression = Expression([data.itos[n] for n in clean_eq_list])

                        try:
                            yhats, eqn_pred = expression.execute(p[z][0])
                            errors.append(criterion(yhats, p[z][1]).cpu().numpy())
                            # print(criterion(yhats, p[z][1]))
                            # print(yhats)
                            # print("Valid Equation")
                        except:
                            # print("Invalid Eqn")
                            errors.append(np.inf)
                        for k, err in enumerate(errors):
                            if np.isnan(err):
                                errors[k] = np.inf

                    # print(errors)
                    num.append(np.mean(errors))
            child.num_fitness = np.mean(num)
            child.sym_fitness = np.mean(sym)

        end = time.time()
        # print("evaluation loop", end - start, flush=True)

        #population selection
        population = []
        use_multi = False
        for parent in parents:
            population.append(parent)
        for child in children:
            population.append(child)

        for b, indiv in enumerate(population):
            if indiv.num_fitness < torch.inf:
                use_multi = True
            else:
                use_multi = False

        if not use_multi:
            print("Used single")
            population = sorted(population, key=lambda individual:individual.sym_fitness)
            parents = population[:num_parents]
        else:
            print("Used Multi")
            parents = survivor_selection(population, num_parents)
            if gen_at_multi < gen:
                gen_at_multi = gen

        #saving fitness over time
        parents = sorted(parents, key=lambda individual:individual.sym_fitness)
        if parents[0].sym_fitness < solution_sym_fitness:
            solution_sym_fitness = parents[0].sym_fitness
            solution_num_fitness = parents[0].num_fitness
            solution = parents[0].encoder

        sym_fitness_over_time[gen] = solution_sym_fitness
        num_fitness_over_time[gen] = parents[0].num_fitness
        print("Generation", gen, "Best sym fitness:", solution_sym_fitness, flush=True)
        print("Generation", gen, "Best num fitness:", solution_num_fitness, flush=True)


    print("best fitness is:", solution_sym_fitness)
    print("Generation for Pareto Front:", gen_at_multi)
    np.save('data/fitovertime'+str(uniqueID), sym_fitness_over_time)
    np.save('data/numfitovertime' + str(uniqueID), num_fitness_over_time)
    torch.save(solution.state_dict(), 'SavedModels/bestencoder'+str(uniqueID))
    for a,parent in enumerate(parents):
        np.save('data/parent'+str(a)+'_symfitness_'+str(uniqueID), parent.sym_fitness)
        np.save('data/parent' + str(a) + '_numfitness_'+str(uniqueID), parent.num_fitness)
        torch.save(parent.encoder.state_dict(), 'SavedModels/'+str(a)+'encoder'+str(uniqueID)+'_best')

