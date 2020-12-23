from EpEnvs.envs.EHEpileptor_Env import EHEpileptorEnv

params1 = {
    "a_1": 1,
    "b_1": 3,
    "c_1": 1,
    "d_1": 5,
    "I_ext1": 3.1,
    "m": 0,
    "a_2": 6,
    "tau_2": 10,
    "I_ext2": .45,
    "gamma": .01,
    "r": .00035,
    "s": 4,
    "x0": -1.6,
    "tstep": .05
}

Env = EHEpileptorEnv(params1)

for i in range(1000):
    print(i)
    print(Env.step(None)[0])
