# -*- coding: utf-8 -*-
"""A random BPMN simulator with given parameter

This simulator class plays out input BPMN model according to given parameter.
The parameters are given with parameter class.

Example
-------

    from src.BPMNSimulator import Simulator


Notes
-----


"""
# import packages
import pandas as pd
from src.SimCore import Process


class BPMNSimulator(object):
    def __init__(self, bpmn_seed, env, para, max_sim_iter=0):
        """
        BPMN simulator class

        Parameters
        ----------
        bpmn_seed
            a string that contains encoding of an BPMN model

        env
            simulation environment

        para
            simulation parameter

        sim_interval
            interval between two consecutive starting of the process

        max_sim_iter
            maximum number of simulation iteration
        """
        self.bpmn_seed = bpmn_seed.split(".")
        self.env = env
        self.para = para
        self.arrival_func = self.para.get_arrival
        self.max_sim_iter = max_sim_iter
        self.sim_iter = 0
        self.forward = env.event()
        self.log = []
        self.env.process(self.start_simulation())

    def parse_bpmn(self, seed) -> list:
        """
        Parse the input BPMN model abstraction in str form and generate an event object list accordingly

        Parameters
        ----------
        seed
            BPMN seed

        Returns
        -------
        list
            event object list contains the sequential ordering of the events
        """
        current_event = None
        following_seq = []
        if len(seed) > 0:
            seed_header = seed[0]

            # generate loop structure
            if seed_header[0] == "l":
                seed_seg = seed_header[1:].split("-")
                node_id1 = seed_seg[-2]
                # node_id2 = seed_seg[-1]

                len_sub1 = int(seed_seg[0])
                seed_sub1 = seed[1:1 + len_sub1]
                sub_event_queue1 = self.parse_bpmn(seed_sub1)

                seed_sub2 = []
                sub_event_queue2 = []
                len_sub2 = 0
                if len(seed_seg) > 3:
                    len_sub2 = int(seed_seg[1])
                    seed_sub2 = seed[1 + len_sub1:1 + len_sub1 + len_sub2]
                    sub_event_queue2 = self.parse_bpmn(seed_sub2)
                gate_para = self.para.get_gate_para(node_id1)
                current_event = Process.LoopEvent(node_id1, node_id1,
                                                  gate_para.get_gate_cond(),
                                                  self.para.scenario,
                                                  sub_event_queue1, sub_event_queue2, "Loop")

                seed = seed[1 + len_sub1 + len_sub2:]

            if seed_header[0] == "p" or seed_header[0] == "x":
                seed_seg = seed_header[1:].split("-")
                len_sub1 = int(seed_seg[0])
                len_sub2 = int(seed_seg[1])
                seed_sub1 = seed[1: 1 + len_sub1]
                seed_sub2 = seed[1 + len_sub1: 1 + len_sub1 + len_sub2]
                node_id1 = seed_seg[-2]
                sub_event_queue1 = self.parse_bpmn(seed_sub1)
                sub_event_queue2 = self.parse_bpmn(seed_sub2)
                gate_para = self.para.get_gate_para(node_id1)
                if seed_header[0] == "p":
                    current_event = Process.AndGateEvent("parallel_" + node_id1, node_id1,
                                                         gate_para.get_gate_cond(),
                                                         self.para.scenario,
                                                         sub_event_queue1, sub_event_queue2, "Parallel")
                if seed_header[0] == "x":
                    current_event = Process.XorGateEvent("exclusive_" + node_id1, node_id1,
                                                         gate_para.get_gate_cond(),
                                                         self.para.scenario,
                                                         sub_event_queue1, sub_event_queue2,
                                                         "Exclusive", gate_para.get_gate_attr())

                seed = seed[1 + len_sub1 + len_sub2:]

            # return sub sequence with exact one task
            if seed_header[0] == "t":
                node_id = seed_header[2:]
                current_event = Process.TaskEvent(node_id, node_id,
                                                  self.para.get_task_para(node_id),
                                                  self.para.scenario)
                seed = seed[1:]

            if len(seed) > 0:
                following_seq = self.parse_bpmn(seed)
            else:
                following_seq = []

        return [current_event, *following_seq]

    def process_level_callback(self, event):
        """
        Event finishing callback prototype

        Parameters
        ----------
        event
            event object which generates this callback
        """
        print('call back from process level finishing event')

    def start_simulation(self):
        """
        Generator for starting simulation
        """

        while True:
            if self.max_sim_iter > 0:
                if self.sim_iter < self.max_sim_iter:
                    self.start_process()
                    #print("test iter, ", self.sim_iter)
                else:
                    break
            else:
                self.start_process()
            yield self.env.timeout(self.para.get_sim_interval())

    def start_process(self):
        """
        Generator for starting sub process
        """
        if self.arrival_func():
            event_queue = self.parse_bpmn(self.bpmn_seed)
            # print("parsed queue: ", event_queue)
            starting_process = Process.ProcessThread(self.env, None, event_queue=event_queue)
            self.env.process(starting_process.operate(self.log))
            self.sim_iter += 1

    def generate_log(self):
        """
        Log the event simulation

        Returns
        -------
        list
            output event log
        """
        if len(self.log) > 0:
            generated_log = pd.DataFrame(data=self.log, columns=["CaseID", "ProcessID", "Activity",
                                                                 "Type", "RegistrationTime", "TaskDuration",
                                                                 "StartTime", "EndTime", "Resource", "Attribute", "GateAttr",
                                                                 "WaitTime"])
            # generated_log["CaseID"] = generated_log["SubprocessID"][0]
            generated_log = generated_log[generated_log["Type"] != "Parallel"]
            generated_log = generated_log[generated_log["Type"] != "Exclusive"]
            generated_log = generated_log[generated_log["Type"] != "Loop"]
            generated_log["TimeLapse"] = generated_log["EndTime"] - generated_log["RegistrationTime"]
            generated_log["ExecutionLapse"] = generated_log["EndTime"] - generated_log["StartTime"]
            generated_log = generated_log.reset_index(drop=True)
            generated_log["RegistrationTime"] = pd.to_datetime(generated_log["RegistrationTime"],
                                                               unit=self.para.time_unit,
                                                               origin=self.para.time_ref)
            generated_log["StartTime"] = pd.to_datetime(generated_log["StartTime"],
                                                        unit=self.para.time_unit,
                                                        origin=self.para.time_ref)
            generated_log["EndTime"] = pd.to_datetime(generated_log["EndTime"],
                                                      unit=self.para.time_unit,
                                                      origin=self.para.time_ref)

            return generated_log[["CaseID", "Activity", "Type", "RegistrationTime",
                                  "StartTime", "EndTime", "TaskDuration", "TimeLapse",
                                  "ExecutionLapse", "GateAttr", "Resource", "WaitTime"]]
        else:
            return []