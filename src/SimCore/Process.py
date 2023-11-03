# -*- coding: utf-8 -*-
"""Process module for BPMN Simulator

This module contains several key components for BPMN Simulator. Events such as tasks, gateways and loop structures are
modeled with corresponding class. A process object can be created using ProcessThread class, which takes a list of event
objects as event queue and then execute them.


Example
-------

    from src.BPMNGenerator import Process


Notes
-----
"""
import uuid
import simpy
import random


class ProcessThread(object):
    """
    Process thread class for creating process running in simulator.

    Attributes
    ----------
    process_id : str
        randomly generated unique id

    case_level : bool
        True if this process is the root process of a case

    process_log : list
        a list of event log data generated when executing the event queue

    Methods
    -------
    operate(log=None)
        generator for executing the event queue
    """
    def __init__(self, env: simpy.Environment, finishing_event: simpy.Event,
                 case_id="", event_queue=[]):
        """

        Parameters
        ----------
        env
            simulation environment

        finishing_event
            event being triggered when the process finished

        case_id
            string id to identify cases

        event_queue
            lists of event objects to be executed in the process
        """
        self.env = env
        random_uuid = uuid.uuid4().hex
        self.process_id = random_uuid[:5]
        self.event_queue = event_queue
        self.finishing_event = finishing_event
        self.case_id = case_id
        self.case_level = False
        self.process_log = []
        if self.finishing_event is None:
            self.finishing_event = env.event()
            self.case_id = random_uuid
            self.case_level = True

    def operate(self, log=None, gate_attr=0, finished_event_num=0):
        """
        Generator for executing the event queue

        Parameters
        ----------
        log
            event log object
        """
        # print("running process")
        attr = gate_attr
        if self.case_level:
            log_temp = self.process_log
        else:
            log_temp = log
        # print("event queue: ", self.event_queue)
        for event in self.event_queue:
            #print("at: ", self.env.now, ' process ', self.process_id, 'case : ', self.case_id, "event: ", event.id)

            event_log = [self.case_id, self.process_id, event.get_name(), event.get_msg(), self.env.now, event.lapse_value]

            event_process = self.env.process(event.operate(self.env, self.case_id, log_temp, finished_event_num))
            yield event_process
            finished_event_num = finished_event_num + 1
            #print("finish event: ", event.get_name())
            # print("subprocess finish")
            # print("finish event: ", event.get_name())
            if log_temp is not None:
                event_attr = event.get_attr()
                if event_attr[-1] > 0:
                    attr = event_attr[-1]
                #print("test gate attr: ", event_attr[-1], " attr:", attr)
                log_temp.append([*event_log, event.execution_time,
                                 event.finish_time, event.res_log, event.get_attr(), attr, event.waiting_time])

        if self.case_level and (log is not None):
            log.extend(self.process_log)
        self.finishing_event.succeed()
        # print("log len: ", len(log))


class Event(object):
    def __init__(self, event_id, event_name, scenario=object(), event_msg=""):
        """
        Base object for different BPMN events

        Parameters
        ----------
        env
            simulation environment

        event_id
            an uuid to identify the event object

        event_name
            event object name

        scenario
            a simulation scenario object that contains environment variables for simulation

        event_msg
            event message
        """
        self.id = event_id
        self.name = event_name
        self.msg = event_msg
        self.scenario = scenario
        self.finish_time = 0
        self.lapse_value = 0
        self.execution_time = 0
        self.attr = 0
        self.waiting_time = 0

    def subprocess_level_callback(self, event):
        """
        Event finishing callback prototype

        Parameters
        ----------
        event
            event object which generates this callback
        """
        # print("call back at: ", event.env.now)
        self.finish_time = event.env.now

    def operate(self, env, case_id, log=None, finished_event_num=0):
        """
        Generator for executing the event

        Parameters
        ----------
        env

        case_id
            string id to identify cases

        log
            event log object
        """
        finished = env.event()
        yield finished.succeed()

    def get_msg(self) -> str:
        """
        Return the event message

        Returns
        -------
        str
            Event message
        """
        return self.msg

    def get_name(self) -> str:
        """
        Return the event name

        Returns
        -------
        str
            Event name
        """
        return self.name

    def get_id(self) -> str:
        """
        Return the event id

        Returns
        -------
        str
            Event id
        """
        return self.id

    def get_attr(self) -> list:
        """
        Get the attribute value of this event

        Returns
        -------
        list
            Event attribute value
        """
        return [self.attr]


class TaskEvent(Event):
    def __init__(self, event_id, event_name, task_para, scenario=object()):
        """
        Task event object

        Parameters
        ----------
        event_id
            an uuid to identify the event object

        event_name
            event object name

        task_para
            task parameter class

        scenario
            a simulation scenario object that contains environment variables for simulation
        """
        super().__init__(event_id, event_name, scenario, "Task")
        self.task_para = task_para
        self.lapse_value = self.task_para.get_single_lapse()
        self.task_func = self.task_para.task_func
        self.resource = self.task_para.resource
        self.res_log = []

    def copy_para(self):
        return self.id, self.name, self.task_para, self.scenario

    def subprocess_level_callback(self, event):
        """
        Task event finishing callback

        Parameters
        ----------
        event
            event object which generates this callback
        """
        # print("call back at: ", event.env.now)
        if self.task_func is not None:
            self.task_func(self.scenario)

    def operate(self, env, case_id, log=None, finished_event_num=0):
        """
        Generator for executing the event

        Parameters
        ----------
        log
        case_id
        env

        """
        finished = env.event()
        time_left = self.lapse_value
        self.res_log = []
        res_in_use = []
        interrupted = 0
        active_resources = []
        while time_left > 0:
            res_in_use = []
            req_process_list = []
            if len(self.resource) > 0:
                granted_req = []
                i = 0
                if len(active_resources) == 0:
                    for res_group in self.resource:
                        res_granted = env.event()

                        priority = res_group.starting_prio - (50*interrupted+finished_event_num) * res_group.incre_prio

                        active_resource_pool = []
                        req_process = env.process(res_group.request_resource_group(finished, res_in_use, res_granted,
                                                                     env.active_process, active_resource_pool,
                                                                     priority=priority))
                        active_resources.append(active_resource_pool)
                        granted_req.append(res_granted)
                        req_process_list.append(req_process)

                else:
                    for res_group in self.resource:
                        res_granted = env.event()

                        priority = res_group.starting_prio - (50*interrupted+finished_event_num) * res_group.incre_prio
                        if i < len(active_resources):
                            req_process = env.process(res_group.request_resource_group(finished, res_in_use, res_granted,
                                                                     env.active_process, active_resources[i],
                                                                     priority=priority))
                        else:
                            active_resource_pool = []
                            req_process = env.process(
                                res_group.request_resource_group(finished, res_in_use, res_granted,
                                                                 env.active_process, active_resource_pool,
                                                                 priority=priority))
                            active_resources.append(active_resource_pool)
                        i = i + 1
                        req_process_list.append(req_process)
                        granted_req.append(res_granted)

                try:
                    yield simpy.events.AllOf(env, granted_req)

                except simpy.Interrupt as interrupt:
                    if len(res_in_use) > 0:
                        for running_res in res_in_use:
                            running_res[0].release(running_res[1])
                    #print("error time: ", env.now)
                    for req_pro in req_process_list:
                        if req_pro.is_alive:
                            req_pro.interrupt()
                    continue

                self.waiting_time = self.waiting_time + res_in_use[-1][-2]

                #print("test res pool: ", active_resources)
                if len(res_in_use) > 0:
                    for running_res in res_in_use:
                        self.res_log.append(running_res[2:-1])
            try:
                if self.execution_time == 0:
                    self.execution_time = env.now
                yield env.timeout(time_left)
                time_left = 0

            except simpy.Interrupt as interrupt:
                usage = env.now - interrupt.cause.usage_since
                time_left -= usage
                if len(res_in_use) > 0:
                    for running_res in res_in_use:
                        running_res[0].release(running_res[1])
                interrupted = interrupted + 1

        if len(res_in_use) > 0:
            for running_res in res_in_use:
                running_res[0].release(running_res[1])

        finished.callbacks.append(self.subprocess_level_callback)
        self.finish_time = env.now
        finished.succeed()


class GateEvent(Event):
    def __init__(self, event_id, event_name, gate_cond, scenario=object(),
                 event_queue1=[], event_queue2=[], event_msg=""):
        """
        Gateway event object

        Parameters
        ----------

        event_id
            an uuid to identify the event object

        event_name
            event object name

        gate_cond
            callable function to determine the gateway condition

        scenario
            a simulation scenario object that contains environment variables for simulation

        event_queue1
            list of event objects in the sub sequence of first branch

        event_queue2
            list of event objects in the sub sequence of second branch

        event_msg
            event message
        """
        super().__init__(event_id, event_name, scenario, event_msg)
        self.gate_cond = gate_cond
        self.event_queue1 = event_queue1
        self.event_queue2 = event_queue2
        self.res_log = []

    def copy_para(self):
        return self.id, self.name, self.gate_cond, self.scenario, self.event_queue1, self.event_queue2, self.msg


class AndGateEvent(GateEvent):
    def __init__(self, event_id, event_name, gate_cond, scenario=object(),
                 event_queue1=[], event_queue2=[], event_msg=""):
        """
        Parallel gateway event

        Parameters
        ----------
        event_id
            an uuid to identify the event object

        event_name
            event object name

        gate_cond
            callable function to determine the gateway condition

        scenario
            a simulation scenario object that contains environment variables for simulation

        event_queue1
            list of event objects in the sub sequence of first branch

        event_queue2
            list of event objects in the sub sequence of second branch

        event_msg
            event message
        """
        super().__init__(event_id, event_name, gate_cond, scenario,
                         event_queue1, event_queue2, event_msg)
        self.attr = 1

    def copy_para(self):
        return self.id, self.name, self.gate_cond, self.scenario, self.event_queue1, self.event_queue2, self.msg

    def operate(self, env, case_id, log=None, finished_event_num=0):
        """
        Generator for executing the event

        Parameters
        ----------
        env
        case_id
            string id to identify the process this task is within

        log
            event log object
        """
        sub_seq1_fin = env.event()
        sub_seq2_fin = env.event()
        finished = env.all_of([sub_seq1_fin, sub_seq2_fin])
        finished.callbacks.append(self.subprocess_level_callback)
        starting_process1 = ProcessThread(env, sub_seq1_fin,
                                          case_id, self.event_queue1)
        starting_process2 = ProcessThread(env, sub_seq2_fin,
                                          case_id, self.event_queue2)

        if random.random() > 0.5:
            env.process(starting_process1.operate(log, self.attr, finished_event_num))
            env.process(starting_process2.operate(log, self.attr, finished_event_num))

        else:
            env.process(starting_process2.operate(log, self.attr, finished_event_num))
            env.process(starting_process1.operate(log, self.attr, finished_event_num))

        yield finished


class XorGateEvent(GateEvent):
    def __init__(self, event_id, event_name, gate_cond, scenario=object(),
                 event_queue1=[], event_queue2=[], event_msg="", gate_attr=0.5):
        """
        Exclusive gateway event

        Parameters
        ----------
        event_id
            an uuid to identify the event object

        event_name
            event object name

        gate_cond
            callable function to determine the gateway condition

        scenario
            a simulation scenario object that contains environment variables for simulation

        event_queue1
            list of event objects in the sub sequence of first branch

        event_queue2
            list of event objects in the sub sequence of second branch

        event_msg
            event message
        """
        super().__init__(event_id, event_name, gate_cond, scenario,
                         event_queue1, event_queue2, event_msg)
        self.gate_attr = gate_attr

    def copy_para(self):
        return self.id, self.name, self.gate_cond, self.scenario, self.event_queue1, self.event_queue2, self.msg, self.gate_attr

    def operate(self, env, case_id, log=None, finished_event_num=0):
        """
        Generator for executing the event

        Parameters
        ----------
        env
        case_id
            string id to identify the process this task is within

        log
            event log object
        """
        finished = env.event()
        sub_seq_queue = self.event_queue1
        while self.attr == 0:
            self.attr = random.uniform(0, self.gate_attr)
        if self.gate_cond(self.scenario):
            sub_seq_queue = self.event_queue2
            while self.attr <= self.gate_attr:
                self.attr = random.uniform(self.gate_attr, 1)
        starting_process = ProcessThread(env, finished, case_id, sub_seq_queue)
        env.process(starting_process.operate(log, self.attr, finished_event_num))
        yield finished


class LoopEvent(XorGateEvent):
    def __init__(self, event_id, event_name, gate_cond, scenario=object(),
                 event_queue1=[], event_queue2=[], event_msg="", gate_attr=0.5):
        """
        Loop event

        Parameters
        ----------
        event_id
            an uuid to identify the event object

        event_name
            event object name

        gate_cond
            callable function to determine the gateway condition

        scenario
            a simulation scenario object that contains environment variables for simulation

        event_queue1
            list of event objects in the sub sequence of mainstream branch

        event_queue2
            list of event objects in the sub sequence of detouring branch

        event_msg
            event message
        """
        super().__init__(event_id, event_name, gate_cond, scenario,
                         event_queue1, event_queue2, event_msg)
        self.process_id = uuid.uuid4().hex[:5]
        self.gate_attr = gate_attr
        self.attr = []

    def copy_para(self):
        return self.id, self.name, self.gate_cond, self.scenario, self.event_queue1, self.event_queue2, self.msg

    def looping(self, env, finishing_event, case_id, log=None, finished_event_num=0):
        """
        Generator for looping

        Parameters
        ----------
        env
        finishing_event
        case_id
            string id to identify the process this task is within

        log
            event log object
        """
        attr = 0
        sub_seq_fin = env.event()
        loop_instance1 = copy_sub_sequence(self.event_queue1)
        starting_process1 = ProcessThread(env, sub_seq_fin, case_id, loop_instance1)
        env.process(starting_process1.operate(log, attr, finished_event_num))
        yield sub_seq_fin
        while self.gate_cond(self.scenario):
            attr = 0
            sub_seq_fin = env.event()
            loop_instance1 = copy_sub_sequence(self.event_queue1)
            loop_instance2 = copy_sub_sequence(self.event_queue2)
            loop_instance = [*loop_instance2, *loop_instance1]
            starting_process2 = ProcessThread(env, sub_seq_fin, case_id, loop_instance)

            while attr == 0:
                attr = random.uniform(0, self.gate_attr)
            self.attr.append(attr)
            env.process(starting_process2.operate(log, attr, finished_event_num))
            yield sub_seq_fin
        while attr <= self.gate_attr:
            attr = random.uniform(self.gate_attr, 1)
        self.attr.append(attr)
        finishing_event.succeed()

    def operate(self, env, case_id, log=None, finished_event_num=0):
        """
        Generator for starting the loop

        Parameters
        ----------
        env

        case_id
            string id to identify the process this task is within

        log
            event log object
        """
        finished = env.event()
        finished.callbacks.append(self.subprocess_level_callback)
        env.process(self.looping(env, finished, case_id, log, finished_event_num))
        yield finished

    def get_attr(self) -> list:
        """
        Get the attribute value of this event

        Returns
        -------
        list
            Event attribute value
        """
        return self.attr


def copy_sub_sequence(event_queue):
    copy_output = []
    for event in event_queue:
        event_type = type(event)
        copy_event = event_type(*event.copy_para())
        copy_output.append(copy_event)
    return copy_output
