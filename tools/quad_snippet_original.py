"""Reference implementation of the 6-DoF quadcopter prototype (verbatim).

Vendored unchanged from the ANN2SNN ``drone-example`` branch
(``tools/quad_snippet_original.py``, MIT; see ``NOTICE``).  It is the standalone
numpy script the ported plant/task are validated against.

    python3 tools/quad_snippet_original.py

Rates are the original: 200 Hz outer (5 ms) with a 10 kHz inner substep over 20 s.
"""

from __future__ import annotations

import numpy as np


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


def rodrigues_exp(w_dt):
    th = np.linalg.norm(w_dt)
    if th < 1e-8:
        return np.eye(3) + skew(w_dt)
    K = skew(w_dt / th)
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def reorthonormalize(R):
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean


class FullPhysicsQuadcopter10kHz:
    def __init__(self):
        self.m, self.g = 0.5, 9.81
        self.J = np.diag([0.0023, 0.0023, 0.0040])
        self.J_inv = np.linalg.inv(self.J)
        self.d = 0.15
        self.d_arm = self.d / np.sqrt(2.0)

        self.rho = 1.225
        self.C_D_body = np.array([0.022, 0.022, 0.048])
        self.D_rot_aero = np.array([0.0012, 0.0012, 0.0025])

        self.C_T = 1.5e-5
        self.C_Q = 2.5e-7
        self.R_prop = 0.0635
        self.k_vz = 1.4e-4
        self.k_h = 1.6e-6

        self.J_m = 1.2e-5
        self.K_t = 0.015
        self.K_e = 0.015
        self.d_m = 1.5e-5
        self.R_coil = 0.18
        self.tau_e = 0.002

        self.capacity_Ah = 1.3
        self.R_int = 0.035
        self.V_min_ocv = 13.2
        self.V_max_ocv = 16.8
        self.I_avionics = 0.25

        c_ratio = self.C_Q / self.C_T
        self.B = np.array([
            [1.0, 1.0, 1.0, 1.0],
            [-self.d_arm, self.d_arm, self.d_arm, -self.d_arm],
            [-self.d_arm, self.d_arm, -self.d_arm, self.d_arm],
            [-c_ratio, -c_ratio, c_ratio, c_ratio],
        ])
        self.B_inv = np.linalg.inv(self.B)

        self.target = np.array([0.0, 0.0, 2.5])
        self.x_obs = np.array([1.35, 0.0, 1.5])
        self.r_core = 0.35
        self.r_ds = 0.62
        self.r_inf = 1.05

        self.heading_d_min = 0.15
        self.heading_d_max = 0.60

    def compute_battery_vocv(self, soc):
        soc = np.clip(soc, 0.0, 1.0)
        return self.V_min_ocv + 3.0 * soc + 0.6 * (soc ** 2)

    def run(self, sim_time=20.0, dt_outer=0.005, f_inner=10000):
        outer_steps = int(sim_time / dt_outer)
        dt_sub = 1.0 / f_inner
        micro_steps = int(dt_outer / dt_sub)

        pos = np.array([-2.2, 0.0, 0.5])
        vel = np.zeros(3)
        R = np.eye(3)
        omega = np.zeros(3)

        omega_m = np.ones(4) * 286.0
        current = np.ones(4) * 1.65
        tau_snn_filtered = np.zeros(3)
        soc = 0.95

        tel = {
            "time": [], "pos": [], "vel": [], "R": [], "omega": [],
            "F_thrust": [], "g_force": [], "V_lyapunov": [], "V_dot": [],
            "advance_ratio": [], "deflection_deg": [], "soc": [],
            "V_term": [], "V_ocv": [], "total_power": [], "motor_rpm": [],
            "motor_currents": [],
        }

        for k in range(outer_steps):
            t = k * dt_outer

            e = pos - self.target
            v_nom = np.array([-0.35 * e[0] - 1.35 * e[1],
                              -0.35 * e[1] + 1.35 * e[0],
                              -0.45 * e[2]])
            speed = np.linalg.norm(v_nom)
            if speed > 2.2:
                v_nom = (v_nom / speed) * 2.2

            dx, dy = pos[0] - self.x_obs[0], pos[1] - self.x_obs[1]
            dist_xy = np.hypot(dx, dy) + 1e-6

            if dist_xy < self.r_inf:
                n = np.array([dx / dist_xy, dy / dist_xy, 0.0])
                t_circ = np.array([-n[1], n[0], 0.0])
                v_perp = v_nom - np.dot(v_nom, n) * n
                t_vec = t_circ if np.dot(v_perp, t_circ) >= 0 else -t_circ

                E = np.column_stack([n, t_vec, [0, 0, 1]])
                gamma = (dist_xy / self.r_ds) ** 2
                lr = 1.0 - (1.0 / max(gamma, 0.05))
                lt = 1.0 + (0.8 / max(gamma, 0.05))
                s = np.clip((self.r_inf - dist_xy) / (self.r_inf - self.r_ds), 0.0, 1.0)
                w = np.sin(np.pi * 0.5 * s) ** 2
                M = np.eye(3) + w * (E @ np.diag([lr, lt, 1.0]) @ E.T - np.eye(3))
                v_des = M @ v_nom
            else:
                v_des = v_nom

            cos_deflect = np.dot(v_nom, v_des) / (
                np.linalg.norm(v_nom) * np.linalg.norm(v_des) + 1e-6)
            deflection_deg = np.degrees(np.arccos(np.clip(cos_deflect, -1.0, 1.0)))

            vh = v_des / (np.linalg.norm(v_des) + 1e-6)
            D = 1.2 * np.outer(vh, vh) + 4.5 * (np.eye(3) - np.outer(vh, vh))
            F_damping = -D @ (vel - v_des)
            F_gravity = np.array([0.0, 0.0, -self.m * self.g])
            F_des = -F_gravity + F_damping

            z_b_des = F_des / (np.linalg.norm(F_des) + 1e-6)
            dist_to_apex = np.linalg.norm(self.target - pos)

            d_min, d_max = self.heading_d_min, self.heading_d_max
            look_dir = (self.target - pos) / dist_to_apex if dist_to_apex > d_min \
                else np.array([1.0, 0.0, 0.0])
            fixed_dir = np.array([1.0, 0.0, 0.0])
            sblend = np.clip((dist_to_apex - d_min) / (d_max - d_min), 0.0, 1.0)
            wblend = sblend * sblend * (3 - 2 * sblend)
            blended = wblend * look_dir + (1.0 - wblend) * fixed_dir
            cam_dir = blended / (np.linalg.norm(blended) + 1e-6)

            y_b_des = np.cross(z_b_des, cam_dir)
            y_b_des /= (np.linalg.norm(y_b_des) + 1e-6)
            x_b_des = np.cross(y_b_des, z_b_des)
            R_d = np.column_stack([x_b_des, y_b_des, z_b_des])

            e_R = 0.5 * vee(R_d.T @ R - R.T @ R_d)
            omega_des = -6.0 * e_R

            raw_snn = -np.array([0.08 * e_R[0], 0.08 * e_R[1], 0.015 * e_R[2]])
            tau_snn_filtered += (raw_snn - tau_snn_filtered) * (dt_outer / 0.015)

            tau_roll = tau_snn_filtered[0] - 0.022 * (omega[0] - omega_des[0])
            tau_pitch = tau_snn_filtered[1] - 0.022 * (omega[1] - omega_des[1])
            tau_yaw = np.clip(tau_snn_filtered[2] - 0.008 * (omega[2] - omega_des[2]),
                              -0.015, 0.015)

            tau_cmd = np.array([tau_roll, tau_pitch, tau_yaw]) + np.cross(omega, self.J @ omega)
            T_cmd = np.clip(np.dot(F_des, R[:, 2]), 1.0, 2.0 * self.m * self.g)

            thrust_targets = self.B_inv @ np.hstack([T_cmd, tau_cmd])
            thrust_targets = 1.35 + 1.15 * np.tanh((thrust_targets - 1.35) / 1.15)

            omega_target = np.sqrt(thrust_targets / self.C_T)
            tau_ff = self.d_m * omega_target + self.C_Q * (omega_target ** 2)
            I_target = np.clip((tau_ff + 0.003 * (omega_target - omega_m)) / self.K_t,
                               0.0, 6.5)

            for _ in range(micro_steps):
                I_total = np.sum(current) + self.I_avionics
                V_ocv = self.compute_battery_vocv(soc)
                V_term = V_ocv - I_total * self.R_int
                soc -= (I_total / (self.capacity_Ah * 3600.0)) * dt_sub

                omega_max_voltage = np.maximum(
                    (V_term - current * self.R_coil) / self.K_e, 50.0)

                current += ((I_target - current) / self.tau_e) * dt_sub

                tau_motor = self.K_t * current
                tau_drag_props = self.d_m * omega_m + self.C_Q * (omega_m ** 2)
                omega_m += ((tau_motor - tau_drag_props) / self.J_m) * dt_sub
                omega_m = np.clip(omega_m, 10.0, omega_max_voltage)

                v_body = R.T @ vel
                v_z_inflow = v_body[2]

                delta_f_inflow = self.k_vz * omega_m * v_z_inflow
                f_actual = np.maximum(self.C_T * (omega_m ** 2) - delta_f_inflow, 0.05)

                F_H_body = -self.k_h * np.sum(omega_m) * np.array([v_body[0], v_body[1], 0.0])
                F_drag_fuselage_body = -0.5 * self.rho * self.C_D_body * (
                    v_body * np.abs(v_body))

                actual_wrench = self.B @ f_actual
                tau_aero = actual_wrench[1:4] - self.D_rot_aero * omega

                tau_gyro = -np.cross(omega, self.J @ omega)
                omega += (self.J_inv @ (tau_aero + tau_gyro)) * dt_sub
                R = R @ rodrigues_exp(omega * dt_sub)

                T_actual = actual_wrench[0]
                F_thrust_world = T_actual * R[:, 2]
                F_aero_drag_world = R @ (F_H_body + F_drag_fuselage_body)
                F_net = F_thrust_world + F_aero_drag_world + F_gravity

                vel = vel + (F_net / self.m) * dt_sub
                pos = pos + vel * dt_sub

            R = reorthonormalize(R)

            g_force = np.linalg.norm(F_thrust_world) / (self.m * self.g)
            e_pos = pos - self.target
            V_lyap = 0.5 * np.dot(e_pos, e_pos)
            V_dot = np.dot(e_pos, vel)

            v_in_plane = np.hypot(v_body[0], v_body[1])
            avg_rpm_rad = np.mean(omega_m)
            advance_ratio = v_in_plane / (avg_rpm_rad * self.R_prop + 1e-6)
            power_watts = I_total * V_term

            tel["time"].append(t)
            tel["pos"].append(pos.copy())
            tel["vel"].append(vel.copy())
            tel["R"].append(R.copy())
            tel["omega"].append(omega.copy())
            tel["F_thrust"].append(F_thrust_world.copy())
            tel["g_force"].append(g_force)
            tel["V_lyapunov"].append(V_lyap)
            tel["V_dot"].append(V_dot)
            tel["advance_ratio"].append(advance_ratio)
            tel["deflection_deg"].append(deflection_deg)
            tel["soc"].append(soc * 100.0)
            tel["V_term"].append(V_term)
            tel["V_ocv"].append(V_ocv)
            tel["total_power"].append(power_watts)
            tel["motor_rpm"].append(omega_m * (60.0 / (2.0 * np.pi)))
            tel["motor_currents"].append(current.copy())

        for key in tel:
            if key != "R":
                tel[key] = np.array(tel[key])
        return tel


def main() -> None:
    sim = FullPhysicsQuadcopter10kHz()
    tel = sim.run(sim_time=20.0, dt_outer=0.005, f_inner=10000)
    peak_g = float(tel["g_force"].max())
    peak_mu = float(tel["advance_ratio"].max())
    mean_vdot = float(np.mean(tel["V_dot"]))
    peak_deflect = float(tel["deflection_deg"].max())
    soc_end = float(tel["soc"][-1])
    mah_used = (0.95 - soc_end / 100.0) * sim.capacity_Ah * 1000.0
    print("\n--- Advanced Dynamics & Stability Report (original prototype) ---")
    print(f"Peak G-Force Load Factor:     {peak_g:.2f} G (Hover = 1.00 G)")
    print(f"Peak Rotor Advance Ratio (μ): {peak_mu:.3f} (Blade advance-to-tip speed)")
    print(f"Peak Billard Deflection:      {peak_deflect:.1f}° (Angular detour around pillar)")
    print(f"Mean Lyapunov Contraction V̇:  {mean_vdot:.3f} m²/s (Strictly negative convergence)")
    print(f"Energy Consumed:              {mah_used:.1f} mAh ({soc_end:.1f}% SoC remaining)")
    print(f"Final position:               {np.round(tel['pos'][-1], 4)}")
    print(f"Final distance to goal:       {float(np.linalg.norm(tel['pos'][-1] - sim.target)):.4f} m")
    print(f"Min distance to obstacle axis:{float(np.min(np.hypot(tel['pos'][:, 0] - sim.x_obs[0], tel['pos'][:, 1] - sim.x_obs[1]))):.4f} m")


if __name__ == "__main__":
    main()
