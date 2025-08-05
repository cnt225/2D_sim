"""
simulation.py - 공통 시뮬레이션 로직
main.py와 record_video.py에서 공유하는 시뮬레이션 코드
"""
import numpy as np
from Box2D.b2 import revoluteJointDef
from robot_simulation.control.policy import potential_field_policy, potential_field_pd_policy, rmp_policy

class RobotSimulation:
    """로봇 시뮬레이션 클래스"""
    
    def __init__(self, world, links, obstacles, target, policy_type="potential_field_pd"):
        self.world = world
        self.links = links
        self.obstacles = obstacles
        self.target = np.array(target)
        self.joints = world.joints
        self.policy_type = policy_type
        
        # 시뮬레이션 상태
        self.prev_end_pos = None
        self.link_lengths = [3.0, 2.5, 2.0]  # 링크 길이
        
        # 물리 설정
        self.TIME_STEP = 1.0/60.0
        self.VEL_ITERS = 10
        self.POS_ITERS = 10
        
    def get_joint_angles(self):
        """현재 조인트 각도 계산 (누적)"""
        q1 = self.joints[0].angle if len(self.joints) > 0 else 0
        q2 = q1 + self.joints[1].angle if len(self.joints) > 1 else q1
        q3 = q2 + self.joints[2].angle if len(self.joints) > 2 else q2
        return [q1, q2, q3]
    
    def compute_jacobian(self):
        """해석적 Jacobian 계산"""
        angles = self.get_joint_angles()
        q1, q2, q3 = angles
        L = self.link_lengths
        
        # Jacobian J2 (2×3) 계산
        J2 = np.zeros((2, 3))
        
        # dP/dq1
        J2[0, 0] = -L[0]*np.sin(q1) - L[1]*np.sin(q2) - L[2]*np.sin(q3)
        J2[1, 0] = L[0]*np.cos(q1) + L[1]*np.cos(q2) + L[2]*np.cos(q3)
        
        # dP/dq2
        J2[0, 1] = -L[1]*np.sin(q2) - L[2]*np.sin(q3)
        J2[1, 1] = L[1]*np.cos(q2) + L[2]*np.cos(q3)
        
        # dP/dq3
        J2[0, 2] = -L[2]*np.sin(q3)
        J2[1, 2] = L[2]*np.cos(q3)
        
        return J2
    
    def get_policy_force(self):
        """선택된 정책에 따라 힘 계산"""
        if self.policy_type == "potential_field_pd":
            return potential_field_pd_policy(self.links, self.target, self.obstacles, self.prev_end_pos)
        elif self.policy_type == "potential_field":
            return potential_field_policy(self.links, self.target, self.obstacles)
        elif self.policy_type == "rmp":
            return rmp_policy(self.links, self.target, self.obstacles)
        else:
            # 기본값
            return potential_field_pd_policy(self.links, self.target, self.obstacles, self.prev_end_pos)
    
    def step(self, debug=False):
        """시뮬레이션 한 스텝 실행"""
        # 1) 정책에서 힘 계산
        F = self.get_policy_force()
        
        # 2) 현재 end-effector 위치 저장
        current_end_pos = np.array(self.links[-1].worldCenter)
        
        # 3) Jacobian 계산
        J2 = self.compute_jacobian()
        
        # 4) 토크 계산 (J^T * F)
        tau = J2.T.dot(F)
        
        # 5) 디버그 출력
        if debug:
            angles = self.get_joint_angles()
            print(f"End-effector: {current_end_pos}, Target: {self.target}, Force: {F}")
            print(f"Joint angles: [{angles[0]:.3f}, {angles[1]:.3f}, {angles[2]:.3f}]")
            print(f"Torques: {tau}")
        
        # 6) 토크 적용
        for j, t in zip(self.joints, tau):
            j.bodyB.ApplyTorque(t, wake=True)
        
        # 7) 물리 스텝
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)
        
        # 8) 다음 프레임을 위해 이전 위치 업데이트
        self.prev_end_pos = current_end_pos.copy()
        
        return {
            'end_pos': current_end_pos,
            'target': self.target,
            'force': F,
            'torques': tau,
            'angles': self.get_joint_angles()
        }
    
    def set_target(self, new_target):
        """타겟 위치 변경"""
        self.target = np.array(new_target)
    
    def set_policy(self, policy_type):
        """정책 변경"""
        self.policy_type = policy_type
        
    def get_end_effector_pos(self):
        """현재 end-effector 위치 반환"""
        return np.array(self.links[-1].worldCenter)
        
    def get_distance_to_target(self):
        """타겟까지의 거리 반환"""
        end_pos = self.get_end_effector_pos()
        return np.linalg.norm(self.target - end_pos)
