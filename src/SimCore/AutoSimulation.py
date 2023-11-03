import random
import pandas as pd
import numpy as np
import scipy
import os
import sys
import yaml
from sklearn import linear_model

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)
from src.SimCore import Resource
from src.SimCore import Scenario


class scenario1(object):
    def __init__(self, parts_num):
        self.parts_num = parts_num


class simulation_meta_para(object):
    def __init__(self, env, seed, yaml_path):
        self.env = env
        self.task_list, self.xor_gate_list, self.and_gate_list, self.loop_list = self.parse_bpmn_seed(seed)
        with open(yaml_path, 'r') as file:
            self.yaml = yaml.safe_load(file)
        self.enable_resource = self.yaml["enable_resource"]
        self.enable_calendar = self.yaml["enable_calendar"]
        self.num_pool_task_ratio = self.yaml["num_pool_task_ratio"]
        self.num_pool_lower = self.yaml["num_pool_lower"]
        self.num_pool_upper = self.yaml["num_pool_upper"]
        self.num_agent_lower = self.yaml["num_agent_lower"]
        self.num_agent_upper = self.yaml["num_agent_upper"]
        self.task_duration_dist = self.yaml["task_duration_dist"]
        self.task_duration_mean_dist = self.yaml["task_duration_mean_dist"]
        self.task_duration_mean_loc = self.yaml["task_duration_mean_loc"]
        self.task_duration_mean_scale = self.yaml["task_duration_mean_scale"]
        self.task_duration_std_dist = self.yaml["task_duration_std_dist"]
        self.task_duration_std_loc = self.yaml["task_duration_std_loc"]
        self.task_duration_std_scale = self.yaml["task_duration_std_scale"]
        self.arrival_rate_dist = self.yaml["arrival_rate_dist"]
        self.arrival_rate_loc = self.yaml["arrival_rate_loc"]
        self.arrival_rate_scale = self.yaml["arrival_rate_scale"]
        self.sim_interval_dist = self.yaml["sim_interval_dist"]
        self.sim_interval_loc = self.yaml["sim_interval_loc"]
        self.sim_interval_scale = self.yaml["sim_interval_scale"]
        self.sim_time_unit = self.yaml["sim_time_unit"]

        if self.task_duration_mean_dist == "uniform":
            self.task_duration_mean_rv = scipy.stats.uniform(loc=self.task_duration_mean_loc,
                                                             scale=self.task_duration_mean_scale)

        if self.task_duration_std_dist == "uniform":
            self.task_duration_std_rv = scipy.stats.uniform(loc=self.task_duration_std_loc,
                                                            scale=self.task_duration_std_scale)

        if self.arrival_rate_dist == "constant":
            self.arrical_rate_rv = Scenario.Constant(value=self.arrival_rate_loc)

        if self.sim_interval_dist == "constant":
            self.sim_interval_rv = Scenario.Constant(value=self.sim_interval_loc)

        if self.sim_interval_dist == "uniform_const":
            rand_value = random.randint(self.sim_interval_loc, self.sim_interval_loc + self.sim_interval_scale)
            self.sim_interval_rv = Scenario.Constant(value=rand_value)

        res_pool_num = 1
        if self.num_pool_task_ratio > 0:
            res_pool_num = int(len(self.task_list) / self.num_pool_task_ratio)
            self.num_pool_upper = len(self.task_list)
        self.num_pool_lower = max(res_pool_num, self.num_pool_lower)

    def parse_bpmn_seed(self, bpmn_seed):
        event_list = bpmn_seed.split(".")
        task_list = []
        xor_gate_list = []
        and_gate_list = []
        loop_list = []
        for event in event_list:
            if event.startswith('t'):
                task_list.append(event[2:])
            if event.startswith('x'):
                xor_gate_list.append(event)
            if event.startswith('p'):
                and_gate_list.append(event)
            if event.startswith('l'):
                loop_list.append(event)

        return task_list, xor_gate_list, and_gate_list, loop_list


def generate_random_res_group(meta_para, multi_stage=True):
    env = meta_para.env
    num_pool_lower = meta_para.num_pool_lower
    num_pool_upper = meta_para.num_pool_upper
    num_agent_lower = meta_para.num_agent_lower
    num_agent_upper = meta_para.num_agent_upper

    num_resource_pools = random.randint(num_pool_lower, num_pool_upper)
    output_res_group_list = []
    cal_resource_groups = []
    base_resource_groups = []

    if meta_para.enable_calendar or multi_stage:
        for i in range(num_resource_pools):
            agent_group = Resource.CalendarResourceGroup(env, "925" + str(i), [540, 480, 420], starting_prio=1000)
            cal_resource_groups.append(agent_group)
        output_res_group_list.append(cal_resource_groups)
    for i in range(num_resource_pools):
        agent_group = Resource.BasicResourceGroup(env, "ResGroup_"+str(i), starting_prio=1000)
        base_resource_groups.append(agent_group)
    output_res_group_list.append(base_resource_groups)


    num_workers_per_res_pool = []
    for i in range(num_resource_pools):
        num_workers_per_res_pool.append(random.randint(num_agent_lower, num_agent_upper))

    num_workers_per_res_pool = np.array(num_workers_per_res_pool)

    j = 0
    for num_worker in num_workers_per_res_pool:
        for res_group in output_res_group_list:
            for i in range(num_worker):
                agent = Resource.BasicResource(env, "worker_" + str(j) + "." + str(i), preemptive=True)
                res_group[j].add_resource(agent)
        j = j + 1

    return output_res_group_list


def generate_sim_para(meta_para,
                      resource_groups, multi_stage=True):
    task_list = meta_para.task_list
    xor_gate_list = meta_para.xor_gate_list
    and_gate_list = meta_para.and_gate_list
    loop_list = meta_para.loop_list

    task_para_list = generate_random_task_para(len(task_list),
                                               meta_para.task_duration_dist,
                                               meta_para.task_duration_mean_rv,
                                               meta_para.task_duration_std_rv,
                                               resource_groups,
                                               enable_res=meta_para.enable_resource,
                                               multi_stage=multi_stage)

    output_sim_para = []
    for single_task_para_list in task_para_list:

        task_para_mapping = {}
        for i in range(len(task_list)):
            task_para_mapping[task_list[i]] = single_task_para_list[i]

        sim_para = Scenario.SimulationParameter(arrival_dist=meta_para.arrical_rate_rv,
                                                task_para=single_task_para_list,
                                                gate_para=[Scenario.GateParameter(Scenario.empty_gate)],
                                                scenario=scenario1(20),
                                                time_unit=meta_para.sim_time_unit,
                                                sim_interval=meta_para.sim_interval_rv)

        sim_para.set_para_mapping(task_para_mapping)
        output_sim_para.append(sim_para)

    return output_sim_para


def alloc_resource_pool(task_num, resource_group_list, multi_res=False):
    output_alloc_res_pool = []
    alloc_res_pool = []
    num_res_group = len(resource_group_list[0])
    res_group_index_list = []
    res_group_list_np = np.array(resource_group_list[0], dtype=object)
    if multi_res:
        for j in range(num_res_group):
            alloc_res_num = np.random.choice(task_num) + 1
            res_group_index = np.random.choice(task_num, alloc_res_num, replace=False)
            res_group_index_list.append(res_group_index)
        res_group_index_list = np.array(res_group_index_list, dtype=object)
        for i in range(task_num):
            res_group_index_sel = res_group_index_list - i
            index = np.squeeze(np.array([np.prod(a, axis=0) for a in res_group_index_sel]))
            resource = res_group_list_np[index == 0]
            if len(resource) == 0:
                resource = np.random.choice(res_group_list_np, np.random.choice(num_res_group) + 1)
            if resource.ndim > 1:
                resource = resource.flatten()
            alloc_res_pool.append(resource)
            output_alloc_res_pool.append(alloc_res_pool)
    else:
        if num_res_group >= task_num:
            resource_index = np.random.choice(np.arange(num_res_group), task_num, replace=False).astype(int)
            for res_groups in resource_group_list:
                res_groups_np = np.array(res_groups, dtype=object)
                alloc_res_pool = res_groups_np[resource_index]
                output_alloc_res_pool.append(alloc_res_pool)
        else:
            excess_res_index = np.random.choice(np.arange(num_res_group), task_num - num_res_group, replace=True).astype(int)
            resource_index = np.concatenate((np.arange(num_res_group), excess_res_index)).astype(int)
            np.random.shuffle(resource_index)
            for res_groups in resource_group_list:
                res_groups_np = np.array(res_groups, dtype=object)
                alloc_res_pool = res_groups_np[resource_index]
                output_alloc_res_pool.append(alloc_res_pool)
    return output_alloc_res_pool


def generate_random_task_para(num, time_dist, time_loc_dist, time_std_dist, resource_group_list,
                              enable_res=True, multi_res=False, multi_stage=True):
    ouput_task_para_list = []
    if enable_res or multi_stage:
        resource_pool = alloc_resource_pool(num, resource_group_list, multi_res)
        resource_pool.append([])
    task_para_list = []
    time_rv_list = []
    for i in range(num):
        time_rv_list = []
        task_func = Scenario.empty_task(None)
        loc = int(abs(time_loc_dist.rvs()))
        scale = int(abs(time_std_dist.rvs()))
        if time_dist == "norm" or multi_stage:
            time_dist_rv = scipy.stats.norm(loc=loc, scale=scale)
            time_rv_list.append(time_dist_rv)
        if time_dist == "constant" or multi_stage:
            time_dist_rv = Scenario.Constant(value=loc)
            time_rv_list.append(time_dist_rv)
        # print(time_loc_dist.rvs(), time_std_dist.rvs())
        if multi_stage:
            stage_task_para_list = []
            for res_pool in resource_pool:
                for time_rv in time_rv_list:
                    if len(res_pool) == 0:
                        stage_task_para_list.append(Scenario.TaskParameter(task_func=task_func,
                                                                     time_dist=time_rv,
                                                                     resource=[]))
                        #print("test")
                    else:
                        stage_task_para_list.append(Scenario.TaskParameter(task_func=task_func,
                                                                     time_dist=time_rv,
                                                                     resource=[res_pool[i]]))
            task_para_list.append(stage_task_para_list)
        else:
            if enable_res:
                task_para_list.append(Scenario.TaskParameter(task_func=task_func,
                                                             time_dist=time_dist_rv,
                                                             resource=[resource_pool[0][i]]))
            else:
                task_para_list.append(Scenario.TaskParameter(task_func=task_func,
                                                             time_dist=time_dist_rv,
                                                             resource=[]))
            ouput_task_para_list.append(task_para_list)

    if multi_stage:
        for j in range(len(task_para_list[0])):
            temp_para_list = []
            for i in range(num):
                temp_para_list.append(task_para_list[i][j])
            ouput_task_para_list.append(temp_para_list)
    return ouput_task_para_list


def predict_time_lag(log, sample_ratio=0.5, max_waiting_time=24*60):
    data_size = log.shape[0]
    sample_num = int(data_size * sample_ratio)
    if sample_num > 1:
        regr = linear_model.LinearRegression()
        time_x = np.arange(sample_num).reshape(-1, 1)
        time_y = (log["WaitTime"]).values[-sample_num:]
        # print("regr data: ", time_y)
        regr.fit(time_x, time_y/max_waiting_time)
        #print("linear regr : ", regr.coef_)
        if regr.coef_ > 1e-1:
            return True
    if sample_num > 1:
        if log["WaitTime"].max() > max_waiting_time:
            return True
    return False


def analyze_log_lag(log, meta_para):
    # print(log)
    task_list = meta_para.task_list
    log["TimeRatio"] = log["TimeLapse"] / log["TaskDuration"]
    log = log.sort_values(by=["EndTime"])
    lag_res_list = []
    for task in task_list:
        task_log = log[log["Activity"] == task]
        # print("TimeRatio:\n", task_log["TimeRatio"])
        lag = predict_time_lag(task_log)
        # print("Lagging: :", lag)
        if lag:
            max_lag_idx = task_log["WaitTime"].idxmax()
            # print("Ratio idxmax: ", max_lag_idx)
            resource_info = pd.DataFrame(task_log.loc[max_lag_idx]["Resource"],
                                         columns=["Time", "Resource", "WaitTime"])
            # print("Lag res:\n", resource_info)
            res_lag = resource_info.loc[resource_info["Time"].idxmax()]["Resource"]
            # print("Lag res:", res_lag)
            lag_res_list.append(res_lag)

    lag_res_list = list(dict.fromkeys(lag_res_list))
    return lag_res_list


def add_res(lag_res_list, res_group, env):
    lag_res_group_idx_list = []
    for lag_res in lag_res_list:
        lag_res_group_idx = int(lag_res.split("_")[1].split(".")[0])
        lag_res_group_idx_list.append(lag_res_group_idx)

    lag_res_group_idx_list = list(dict.fromkeys(lag_res_group_idx_list))

    for lag_res_group_idx in lag_res_group_idx_list:
        lag_res_group = res_group[lag_res_group_idx]
        # print("lag_res_group before: ", len(lag_res_group.get_resources()))
        res_num = len(lag_res_group.get_resources())
        agent = Resource.BasicResource(env, "worker_" + str(lag_res_group_idx) + "." + str(res_num + 1),
                                       preemptive=True)
        lag_res_group.add_resource(agent)


def update_res_pool_setting(env, res_groups):
    for res_group in res_groups:
        res_group.set_env(env)
    res_groups = list(dict.fromkeys(res_groups))
    for res_group in res_groups:
        if isinstance(res_group, Resource.CalendarResourceGroup):
            res_group.start_rotate()
    return res_groups


