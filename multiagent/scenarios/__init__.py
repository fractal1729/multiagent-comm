import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)

def make_scenario(scenario_name):
	scenario_name = scenario_name.split('-')
	pathname = osp.join(osp.dirname(__file__), scenario_name[0] + ".py")
	src = imp.load_source('', pathname)
	return src.Scenario(*scenario_name[1:])