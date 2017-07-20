from numpy import array as arr
import program
from utilities import functions
from algorithms import filter_linesearch





from utilities import sys_utils

# sys_utils.findPrintStatements()




import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    #log = file if hasattr(file,'write') else sys.stderr
    log = sys.stderr
    traceback.print_stack(file=log)
    log.write(warnings.formatwarning(message, category, filename, lineno, line))



Q = arr([[1, 0],[0, 5]])
b = arr([0, 0])
#b = arr([1, 1])
c=0


# write tests

q = functions.Quadratic(Q, b, c)
c1 = functions.Line(arr([1,  1]), -1)
#c2 = functions.Line(arr([1, -1]), 10)
c2 = functions.Wiggles()

equality = []
equality.append(c1)

inequality = []
inequality.append(c2)

x0 = arr([1, 2])

warnings.showwarning = warn_with_traceback
statement = program.Program(q, equality, inequality, x0)
#statement = program.DfoProgram("wiggles", q, equality, inequality, x0, plotImprovements=True)
constants = filter_linesearch.Constants(50 * filter_linesearch.theta(statement, x0))
#constants.plot = False

#import cProfile
#tot = cProfile.run('filter_linesearch.filter_line_search(statement, constants)')

results = filter_linesearch.filter_line_search(statement, constants)


print("number_of_iterations                      = " + str(results.number_of_iterations                     ))
print("restorations                              = " + str(results.restorations                             ))
print("ftype_iterations                          = " + str(results.ftype_iterations                         ))
print("filter_modified_count                     = " + str(results.filter_modified_count                    ))
print("pareto                                    = " + str(results.pareto                                   ))
print("success                                   = " + str(results.success                                  ))
print("f_min                                     = " + str(results.f_min                                    ))
print("x_min                                     = " + str(results.x_min                                    ))
print("filterRejectedCount                       = " + str(results.filterRejectedCount                      ))
print("criteria_satifisfied_but_trust_region_not = " + str(results.criteria_satifisfied_but_trust_region_not))
