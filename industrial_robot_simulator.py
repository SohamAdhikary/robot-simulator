# industrial_robot_simulator.py

import pybullet as p
import time
import pybullet_data
import os
import json
from mistral_control import ask_mistral
import subprocess
import numpy as np
from datetime import datetime

class IndustrialRobotSimulator:
    def __init__(self):
        self.physicsClient = None
        self.robot_id = None
        self.box_id = None
        self.gripper_ids = []
        self.conveyor_parts = []
        self.target_id = None
        self.joints = {}
        self.metrics = {
            'success_count': 0,
            'fail_count': 0,
            'steps_to_complete': [],
            'energy_used': 0,
            'task_phases': ["Approach", "Grip", "Move", "Release", "Return"],
            'current_phase': 0,
            'start_time': None,
            'end_time': None
        }

    def initialize_simulation(self):
        try:
            self.physicsClient = p.connect(p.GUI)
            print("PyBullet GUI connected successfully")
        except:
            try:
                self.physicsClient = p.connect(p.DIRECT)
                print("PyBullet DIRECT connected (no visuals)")
            except Exception as e:
                print(f"Failed to connect: {e}")
                exit(1)

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

        p.addUserDebugLine([0,0,0], [1,0,0], [1,0,0], 2)
        p.addUserDebugLine([0,0,0], [0,1,0], [0,1,0], 2)
        p.addUserDebugLine([0,0,0], [0,0,1], [0,0,1], 2)

    def setup_environment(self):
        p.loadURDF("plane.urdf")
        self.create_conveyor_belt()
        p.loadURDF("table/table.urdf", [0.8, 0, 0])

        self.target_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.01], rgbaColor=[0, 1, 0, 0.5])
        p.createMultiBody(baseVisualShapeIndex=self.target_id, basePosition=[0.8, 0, 0.75])

        p.addUserDebugText("TARGET ZONE", [0.8, 0, 0.8], textColorRGB=[0,1,0], textSize=1.2)
        p.addUserDebugText("CONVEYOR", [0, 1.5, 0.5], textColorRGB=[1,1,0], textSize=1.2)

    def create_conveyor_belt(self):
        base_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.1, 0.05])
        base_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.1, 0.05], rgbaColor=[0.4, 0.4, 0.4, 1])
        base_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_shape, baseVisualShapeIndex=base_visual, basePosition=[0, 1.5, 0.05])

        belt_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.09, 0.02])
        belt_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.09, 0.02], rgbaColor=[0.8, 0.8, 0.8, 1])
        belt_id = p.createMultiBody(baseMass=0.1, baseCollisionShapeIndex=belt_shape, baseVisualShapeIndex=belt_visual, basePosition=[0, 1.5, 0.1])

        # Just visually simulate the conveyor — motor control removed to prevent error
        self.conveyor_parts = [base_id, belt_id]

    def spawn_box(self, position):
        box_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1])
        box_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1], rgbaColor=[0.8, 0.2, 0.2, 1])
        return p.createMultiBody(baseMass=1, baseCollisionShapeIndex=box_shape, baseVisualShapeIndex=box_visual, basePosition=position)

    def load_robot(self):
        robots_to_try = ["kuka_iiwa/model.urdf", "ur5/ur5.urdf", "franka_panda/panda.urdf"]
        for robot_file in robots_to_try:
            try:
                robot_id = p.loadURDF(robot_file, [0, 0, 0.5])
                if robot_id != -1:
                    print(f"Loaded robot: {robot_file}")
                    self.robot_id = robot_id
                    self.initialize_gripper()
                    self.initialize_joints()
                    return True
            except Exception as e:
                print(f"Failed to load {robot_file}: {e}")
                continue
        print("ERROR: No suitable robot model found")
        return False

    def initialize_gripper(self):
        left_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05,0.02,0.1])
        left_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.02,0.1], rgbaColor=[0.3,0.3,0.8,1])
        left_gripper = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=left_shape, baseVisualShapeIndex=left_visual, basePosition=[0,0.1,0.7])

        right_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.05,0.02,0.1])
        right_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.05,0.02,0.1], rgbaColor=[0.3,0.3,0.8,1])
        right_gripper = p.createMultiBody(baseMass=0.3, baseCollisionShapeIndex=right_shape, baseVisualShapeIndex=right_visual, basePosition=[0,-0.1,0.7])

        self.gripper_ids = [left_gripper, right_gripper]

        for i, gripper_id in enumerate(self.gripper_ids):
            p.createConstraint(self.robot_id, -1, gripper_id, -1, p.JOINT_POINT2POINT, [0,0,0], [0,0.1 if i==0 else -0.1, 0], [0,0,0])
        print("Gripper initialized")

    def initialize_joints(self):
        self.joints = {}
        for i in range(p.getNumJoints(self.robot_id)):
            info = p.getJointInfo(self.robot_id, i)
            joint_name = info[1].decode('utf-8')
            self.joints[joint_name] = {
                'index': i,
                'type': info[2],
                'lower_limit': info[8],
                'upper_limit': info[9],
                'max_force': info[10],
                'max_velocity': info[11]
            }
            print(f"Discovered joint: {joint_name} (Index: {i})")

    def control_gripper(self, action):
        target_pos = [0, 0.05] if action == "close" else [0, 0.15]
        for i, gripper_id in enumerate(self.gripper_ids):
            p.setJointMotorControl2(gripper_id, 0, p.POSITION_CONTROL, targetPosition=target_pos[i], force=10)

    def check_grasp(self):
        if not self.box_id or not self.gripper_ids:
            return False
        contacts = []
        for gripper_id in self.gripper_ids:
            contacts += p.getContactPoints(self.box_id, gripper_id)
        return len(contacts) >= 2

    def check_task_completion(self):
        if not self.box_id:
            return False
        box_pos, _ = p.getBasePositionAndOrientation(self.box_id)
        target_pos = [0.8, 0, 0.75]
        distance = np.linalg.norm(np.array(box_pos) - np.array(target_pos))
        return distance < 0.2

    def update_task_phase(self):
        if not self.box_id:
            return
        box_pos, _ = p.getBasePositionAndOrientation(self.box_id)
        if self.metrics['current_phase'] == 0:
            if np.linalg.norm(np.array(box_pos) - np.array([0.5,0,0.5])) < 0.3:
                self.metrics['current_phase'] = 1
                self.control_gripper("close")
        elif self.metrics['current_phase'] == 1:
            if self.check_grasp():
                self.metrics['current_phase'] = 2
        elif self.metrics['current_phase'] == 2:
            if self.check_task_completion():
                self.metrics['current_phase'] = 3
                self.control_gripper("open")
                self.metrics['success_count'] += 1
                self.metrics['steps_to_complete'].append(self.current_step)
        elif self.metrics['current_phase'] == 3:
            if not self.check_grasp():
                self.metrics['current_phase'] = 4
        elif self.metrics['current_phase'] == 4:
            if np.linalg.norm(np.array(box_pos) - np.array([0.5,0,0.5])) < 0.3:
                self.reset_task()

    def reset_task(self):
        p.removeBody(self.box_id)
        self.box_id = self.spawn_box([0, 1.5, 0.5])
        self.metrics['current_phase'] = 0

    def get_ai_command(self):
        phase_name = self.metrics['task_phases'][self.metrics['current_phase']]
        available_joints = "\n".join([f"- {name}" for name in self.joints.keys()])
        prompt = f"""
INDUSTRIAL ROBOT CONTROL SYSTEM
CURRENT PHASE: {phase_name}
AVAILABLE JOINTS:
{available_joints}

INSTRUCTIONS:
1. Choose ONE action to {phase_name.lower()} the box:
   - rotate_joint JOINT_NAME clockwise
   - rotate_joint JOINT_NAME counterclockwise
   - no_action
2. Use 5-10 degree steps
3. Respond ONLY with the command
4. Prioritize base joints for gross movement

EXAMPLE: 'rotate_joint lbr_iiwa_joint_3 clockwise'
YOUR RESPONSE: """
        try:
            response = ask_mistral(prompt).strip()
            if response.startswith("rotate_joint"):
                parts = response.split()
                if len(parts) == 3 and parts[1] in self.joints:
                    return response
            return "no_action"
        except Exception as e:
            print(f"AI command failed: {e}")
            return "no_action"

    def execute_command(self, command):
        if command == "no_action":
            return
        parts = command.split()
        joint_name = parts[1]
        direction = parts[2]
        if joint_name not in self.joints:
            print(f"Invalid joint: {joint_name}")
            return
        degrees = 5 if direction == "clockwise" else -5
        joint_info = self.joints[joint_name]
        current_pos = p.getJointState(self.robot_id, joint_info['index'])[0]
        target_rad = current_pos + degrees * (np.pi/180)
        if joint_info['lower_limit'] < joint_info['upper_limit']:
            target_rad = np.clip(target_rad, joint_info['lower_limit'], joint_info['upper_limit'])
        self.metrics['energy_used'] += abs(degrees) * 0.01 * joint_info['max_force']
        p.setJointMotorControl2(
            self.robot_id,
            joint_info['index'],
            p.POSITION_CONTROL,
            targetPosition=target_rad,
            force=joint_info['max_force'],
            maxVelocity=joint_info['max_velocity']
        )

    def save_metrics(self):
        self.metrics['end_time'] = datetime.now().isoformat()
        self.metrics['duration'] = time.time() - self.metrics['start_time']
        self.metrics['success_rate'] = (
            self.metrics['success_count'] / 
            (self.metrics['success_count'] + self.metrics['fail_count'])
            if (self.metrics['success_count'] + self.metrics['fail_count']) > 0 
            else 0
        )

# ------------------------ MAIN EXECUTION ------------------------

if __name__ == "__main__":
    sim = IndustrialRobotSimulator()
    sim.initialize_simulation()
    sim.setup_environment()

    if sim.load_robot():
        sim.box_id = sim.spawn_box([0, 1.5, 0.5])
        sim.metrics['start_time'] = time.time()

        for step in range(1000):
            sim.current_step = step
            sim.update_task_phase()
            command = sim.get_ai_command()
            sim.execute_command(command)
            p.stepSimulation()
            time.sleep(1.0 / 240.0)

        sim.save_metrics()
        p.disconnect()
    else:
        print("Robot failed to load. Exiting.")
