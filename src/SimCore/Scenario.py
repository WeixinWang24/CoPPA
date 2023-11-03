# -*- coding: utf-8 -*-
"""A random BPMN generator with given parameter

This generator class creates randomized BPMN model according to given parameter.
The parameters are given when initializing the generator class.
Run the member function generate() to create the BPMNs.

Example
-------

    from src.BPMNGenerator import Generator


Notes
-----


"""
import random
import scipy
import pandas as pd
import numpy as np


def empty_task(scenario: object):
    """
    Default callable object for tasks which does nothing

    Parameters
    ----------
    scenario
        object that contains the environmental variable for simulation
    """
    pass


def empty_gate(scenario: object):
    """
    Default callable object for gateways to determine the outcome, randomly outputs ture or false

    Parameters
    ----------
    scenario
        object that contains the environmental variable for simulation

    Returns
    -------
    bool
        outcome indicator
    """
    if random.random() < 0.5:
        return True
    else:
        return False


class Constant(scipy.stats.rv_continuous):
    def __init__(self, value=1):
        """
        Random variable object which only outputs constant values

        Parameters
        ----------
        value
            constant output value
        """
        super().__init__()
        self.value = value

    def _rvs(self, loc=0, scale=1, size=1, random_state=0):
        """
        Output constant values

        Parameters
        ----------
        loc
            mean, no use

        scale
            std, no use

        size
            dimension

        random_state
            no use

        Returns
        -------
        nparray
            constant output values
        """
        return np.ones(size) * self.value


class TaskParameter(object):
    def __init__(self, task_func, time_dist=Constant(), lower_bound=1, resource=[]):
        """
        Parameters class for task events used in the simulator

        Parameters
        ----------
        task_func
            a callable object that will be called when the task finishes

        time_dist
            a scipy rv object that describes the distribution of the task's time lapse

        lower_bound
            lower bound of the task's time lapse

        resource
            list of the resource objects which the task requires before it starts
        """
        self.time_lapse_dist = time_dist
        self.lower_bound = lower_bound
        self.resource = resource
        for res_group in self.resource:
            res_group.initialize_group()
        self.task_func = task_func

    def get_single_lapse(self):
        """
        Generate a time lapse value for the task

        Returns
        -------
        int
            time lapse value
        """
        value = self.time_lapse_dist.rvs()
        if value < self.lower_bound:
            value = self.lower_bound
        return int(value)

    def get_task_func(self):
        """
        Get the task function callable object

        Returns
        -------
        callable
            task function callable object
        """
        return self.task_func


class GateParameter(object):
    def __init__(self, gate_cond, gate_attr=0.5):
        """
        Parameters class for gateway used in the simulator

        Parameters
        ----------
        gate_cond
            callable object for gateways to determine the outcome

        """
        self.gate_cond = gate_cond
        self.gate_attr = gate_attr

    def get_gate_cond(self):
        """
        Get the gateway condition callable object

        Returns
        -------
        callable
            gateway condition callable object

        """
        return self.gate_cond

    def get_gate_attr(self):
        """
        Get the gate attribute value which indicates the gateway outcome

        Returns
        -------
        float
            gate attribute value
        """
        return self.gate_attr


class SimulationParameter(object):
    def __init__(self, arrival_dist=Constant(),
                 task_para=[TaskParameter(task_func=empty_task,
                                          time_dist=scipy.stats.norm(loc=5, scale=1))],
                 gate_para=[GateParameter(empty_gate)],
                 scenario=object(),
                 time_ref=pd.Timestamp('2022-01-01'),
                 time_unit="s",
                 sim_interval=Constant(),
                 resource_groups=[]):
        """
        Parameters class contains all the variable for simulation

        Parameters
        ----------
        arrival_dist
            scipy rv object that outputs random values according to its initialization, this random values indicates the
            binomial probability of whether a new simulation iteration starts

        task_para
            list of task parameters which will be assigned to the task events in simulation

        gate_para
            list of gateway parameters which will be assigned to the gateways in simulation

        scenario
            object that contains the environmental variable for simulation

        time_ref
            pandas timestamp object that servers as the start timestamp of the simulation

        time_unit
            unit of the timestamp

        sim_interval
            scipy rv object that outputs random values according to its initialization, this random values indicates the
            interval of possible new simulation arrival

        """
        self.mapping = {}
        self.arrival_dist = arrival_dist
        self.uniform = scipy.stats.uniform()
        self.task_para = task_para
        self.gate_para = gate_para
        self.time_ref = time_ref
        self.time_unit = time_unit
        self.scenario = scenario
        self.sim_interval = sim_interval
        self.resource_groups = resource_groups

    def set_para_mapping(self, mapping):
        """
        Map event parameter with individual event (task/gate)

        Parameters
        ----------
        mapping
            dict object where key refers to event id and value refers to corresponding parameter
        """
        self.mapping = mapping

    def get_task_para(self, task=""):
        """
        Assign task parameter to individual task

        Parameters
        ----------
        task
            task id

        Returns
        -------
        TaskParameter
            assigned task parameter

        """
        default_para = random.choice(self.task_para)
        return self.mapping.get(task, default_para)

    def get_gate_para(self, gate=""):
        """
        Assign gateway parameter to individual gateway

        Parameters
        ----------
        gate
            gate id

        Returns
        -------
        GateParameter
            assigned gate parameter

        """
        default_para = random.choice(self.gate_para)
        return self.mapping.get(gate, default_para)

    def get_arrival(self):
        """
        Determine whether to start a new simulation

        Returns
        -------
        bool
            bool value which determine to start new simulation
        """
        if self.uniform.rvs() < self.arrival_dist.rvs():
            return True
        else:
            return False

    def get_sim_interval(self):
        """
        Generate an interval between two possible new simulations according to distribution

        Returns
        -------
        int
            simulation interval
        """
        interval = self.sim_interval.rvs()
        if interval < 1:
            interval = 1
        return int(interval)