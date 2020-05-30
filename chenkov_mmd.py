# -*- coding: utf-8 -*-
"""
chenkov_mmd.py
Marcello DiStasio, May 2020
based on:

Chenkov et. al. Memory replay in balanced recurrent networks. January 30, 2017. PLOS Computational Biology. 
https://doi.org/10.1371/journal.pcbi.1005359

Model implemented in BRIAN2 (Goodman DFM, Brette R. The brian simulator. Front Neurosci. 2009;3:192–197. pmid:20011141)
https://brian2.readthedocs.io/en/stable/ 

"""

from brian2 import *

import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage

import os, time, datetime
import csv

try:
    clear_cache('cython')
except Exception: 
    pass

fnamebase = os.path.splitext(os.path.basename(__file__))[0] #Basepath for data files output; defaults here to name of python script

# ###########################################
# Defining network model parameters
# ###########################################

NE = 20000          # Number of excitatory cells
NI = int(NE/4)          # Number of inhibitory cells


Nassembly=10   # number of cell assemblies
NEin1assem=400  # number of excitatory cells in an assembly
NIin1assem=int(NEin1assem/4)  # number of inhibitory cells in an assembly

tau_E = 5.0*ms   # Glutamatergic synaptic time constant
tau_gaba = 10.0*ms  # GABAergic synaptic time constant

epsilon = 0.01      # probability of synaptic connections

epsilon_ee = Nassembly/NE      # probability of synaptic connections
epsilon_ee = 0.01              # probability of synaptic connections

prc=.1   #recurrent connection probability <-- 0.1 works!
pff=.01   #feedforward connection probability <-- 0.01 works!

tau_stdp = 20*ms    # STDP time constant #orig 20*ms

# ###########################################
# Simulation epochs
# ###########################################

balancingtime=10*second
runtime = 60*second # Simulation time
testtime = 0.5*second

# ###########################################
# Neuron model
# ###########################################

gLeak = 10.0*nsiemens   # Leak conductance
vRest = -60*mV          # Resting potential
vI = -80*mV             # Inhibitory reversal potential
vE = 0*mV               # Excitatory reversal potential
vt = -50.*mV            # Spiking threshold
memc = 200.0*pfarad     # Membrane capacitance
iExt = 200*pA           # External current
syn_delay=2*ms          # delay from pre-syn to post-syn



eqs_neurons='''
dv/dt=(gLeak*(vRest-v)+gE*(vE-v)+gI*(vI-v)+iExt)/memc : volt (unless refractory)
dgE/dt = -gE/tau_E : siemens
dgI/dt = -gI/tau_gaba : siemens
'''


for EC_input_on in (True, False):


    start_scope() # reset simulation

    # ###########################################
    # Initialize neuron group
    # ###########################################

    neurons = NeuronGroup(NE+NI, model=eqs_neurons, threshold='v > vt',
                          reset='v=vRest', refractory=2*ms, method='euler')
    Pe = neurons[:NE]
    Pi = neurons[NE:]

    # ###########################################
    # Connecting the network
    # ###########################################

    con_e = Synapses(Pe, neurons, on_pre='gE += 0.1*nS',delay=syn_delay)
    con_e.connect(p=epsilon)

    # con_e = Synapses(Pe, Pi, on_pre='gE += 0.1*nS',delay=syn_delay)
    # con_e.connect(p=epsilon)

    con_ii = Synapses(Pi, Pi, on_pre='gI += 0.4*nS',delay=syn_delay)
    con_ii.connect(p=epsilon)


    # ###########################################
    # Excitatory-Excitatory Plasticity
    # ###########################################
    tau_stdp_ee = 200*ms    # STDP time constant #orig 20*ms
    eqs_stdp_ee = '''
    w : 1
    dApre/dt=-Apre/tau_stdp_ee : 1 (event-driven)
    dApost/dt=-Apost/tau_stdp_ee : 1 (event-driven)
    '''
    eqs_stdp_ee_onpre='''Apre += 1.
    w = clip(w+(Apost-alpha)*eta, 0, gmax_ee)
    gE += w*nS'''
    eqs_stdp_ee_onpost='''Apost += 1.
    w = clip(w+Apre*eta, 0, gmax_ee)
    '''
    alpha = 5*Hz*tau_stdp*2  # Target rate parameter
    gmax_ee = 3            # Maximum excitatory-excitatory weight

    # Completely dense connectivity
    # con_ee = Synapses(Pe, Pe, model=eqs_stdp_ee,
    #                   on_pre = eqs_stdp_ee_onpre,
    #                   on_post = eqs_stdp_ee_onpost, delay=syn_delay, )
    # con_ee.connect(p=epsilon_ee, skip_if_invalid=True)


    
    # ###########################################
    # Inhibitory Plasticity
    # ###########################################
    eqs_stdp_inhib = '''
    w : 1
    dApre/dt=-Apre/tau_stdp : 1 (event-driven)
    dApost/dt=-Apost/tau_stdp : 1 (event-driven)
    '''
    eqs_stdp_onpre='''Apre += 1.
    w = clip(w+(Apost-alpha)*eta, 0, gmax)
    gI += w*nS'''
    eqs_stdp_onpost='''Apost += 1.
    w = clip(w+Apre*eta, 0, gmax)
    '''

    #    alpha = 5*Hz*tau_stdp*2  # Target rate parameter
    gmax = 3               # Maximum inhibitory weight
    con_ie = Synapses(Pi, Pe, model=eqs_stdp_inhib,
                      on_pre=eqs_stdp_onpre,
                      on_post=eqs_stdp_onpost,delay=syn_delay)
    con_ie.connect(p=epsilon)

    #con_ie.w = 1e-10
    # ##########################################
    # cell assembly
    # #########################################

    for m in range(Nassembly):

        # recurrent connection
        exec('PeA%s=neurons[m*NEin1assem:(m+1)*NEin1assem]'%(m))
        exec('PiA%s=neurons[NE+m*NIin1assem:NE+(m+1)*NIin1assem]'%(m))

        exec("con_eeA%s = Synapses(PeA%s, PeA%s, on_pre='gE += 0.1*nS',delay=syn_delay)"%(m,m,m))
        exec('con_eeA%s.connect(p=prc)'%(m))
        exec("con_eiA%s = Synapses(PeA%s, PiA%s, on_pre='gE += 0.1*nS',delay=syn_delay)"%(m,m,m))
        exec("con_eiA%s.connect(p=prc)"%(m))

        exec("con_iiA%s = Synapses(PiA%s, PiA%s, on_pre='gI += 0.4*nS',delay=syn_delay)"%(m,m,m))
        exec("con_iiA%s.connect(p=prc)"%(m))
        exec("con_ieA%s = Synapses(PiA%s, PeA%s, model=eqs_stdp_inhib,on_pre=eqs_stdp_onpre,on_post=eqs_stdp_onpost,delay=syn_delay)"%(m,m,m))
        exec("con_ieA%s.connect(p=prc)"%(m))
        exec("con_ieA%s.w = 1e-10"%(m))

        # feedforward connection
        if m>0:
#            exec("con_ff%s=Synapses(PeA%s,PeA%s,on_pre='gE += 0.1*nS',delay=syn_delay)"%(m,m-1,m))
            exec("con_ff%s=Synapses(PeA%s,PeA%s,model=eqs_stdp_ee,on_pre = eqs_stdp_ee_onpre,on_post = eqs_stdp_ee_onpost, delay=syn_delay)"%(m,m-1,m))
            exec("con_ff%s.connect(p=pff)"%(m))



    # ####################################################
    # INPUTS (stimulation, EC, etc.)
    # ####################################################

    # --- Modeling a behavior ---
    # Rat running along linear track in sinusoid

    tmin = balancingtime + testtime
    tmax = balancingtime + testtime + runtime
    dt_inp = defaultclock.dt
    t_inp = arange(tmin,tmax,dt_inp)

    xmax = 1*meter
    dx_inp = 1.0*mm

    # location along track (fraction of track length)
    x_rat = (xmax*(np.sin((t_inp/dt_inp)/(0.4/dt_inp*second))+1)/2)  # Sine wave
    x_rat = xmax*(((t_inp/(1*second)) - np.floor((1/2 + t_inp/(1*second))))+0.5) # Sawtooh

    # --- Modeling entorhinal cortex input ---
    # Gaussian bump centered at fractions of distance along track

    ec_centers = np.linspace(0.001,1,Nassembly)*meter

    ec_rates = np.zeros([len(ec_centers), int(xmax/dx_inp)])
    ec_out = np.zeros([len(ec_centers), int((tmax-tmin)/dt_inp)])
    for cc in range(len(ec_centers)):
        c = signal.unit_impulse(int(xmax/dx_inp), idx=int(ec_centers[cc]*xmax/dx_inp)-1)
        filt = signal.gaussian(1401, std=350)
        ec_rates[cc,:] = signal.convolve(c, filt, mode='same')

        ec_out[cc,:] = ec_rates[cc,np.intp(x_rat/dx_inp)]

    # EC output is zero during balancing (For now)
    EC_max_firing_rate = 100*Hz
    if EC_input_on:
        print("EC real input is ON.")
        EC_rate = np.vstack((np.zeros([np.intp((balancingtime+testtime)/dt_inp),len(ec_centers)]), ec_out.T)) * EC_max_firing_rate * defaultclock.dt
        EC_act = np.random.uniform(size=np.shape(EC_rate)) < EC_rate # This is Poisson
    else:
        print("EC real input is OFF.")
        EC_rate = np.vstack((np.zeros([np.intp((balancingtime+testtime)/dt_inp),len(ec_centers)]), mean(ec_out)*np.ones(np.shape(ec_out.T)) )) * EC_max_firing_rate * defaultclock.dt
        EC_act = np.random.uniform(size=np.shape(EC_rate)) < EC_rate # This is Poisson
    
    [stst, spike_indices] = np.where(EC_act)
    spike_times = stst*defaultclock.dt

    EC = SpikeGeneratorGroup(Nassembly, spike_indices, spike_times, dt=defaultclock.dt)

    # Set up EC input to hippocampus synapses
    p_Input = 0.005
    for m in range(Nassembly):
        #  connection
        exec('con_stimulus%s=Synapses(EC,PeA%s,on_pre=\'gE += 3*nS\',delay=syn_delay)'%(m,m))
        exec('con_stimulus%s.connect(i=%s, j=np.intp(np.arange(len(PeA%s))*np.uint(np.random.random(len(PeA%s))>(1-p_Input))))'%(m,m,m,m))



    # ##########################
    # Set up input for testing with pulse stimulus
    # ##########################

    # after balancing
    st_ind=range(NEin1assem)
    st_time=(balancingtime)*ones(NEin1assem)
    st=SpikeGeneratorGroup(NEin1assem,st_ind,st_time)
    con_stimulus_test=Synapses(st,PeA0,on_pre='gE += 3*nS',delay=syn_delay)
    con_stimulus_test.connect(j='i')

    # after rat running
    st_ind=range(NEin1assem)
    st_time=(balancingtime+testtime+runtime)*ones(NEin1assem)
    st2=SpikeGeneratorGroup(NEin1assem,st_ind,st_time)
    con_stimulus_test2=Synapses(st2,PeA0,on_pre='gE += 3*nS',delay=syn_delay)
    con_stimulus_test2.connect(j='i')



    # ###########################################
    # Setting up monitors
    # ###########################################
    sm = SpikeMonitor(Pe)
    sm2 = SpikeMonitor(EC)

    net=Network(collect())

    # snapshot the state
    net.store('initial')



    # ####################################################################################################
    # -------------------- RUNNING THE SIMULATION --------------------------------------------------------
    # ####################################################################################################


    # --------------------------------------------------
    # Balancing time with plasticity
    # --------------------------------------------------

    print('balancing the network')
    tic=time.time()
    eta = 0.0005          # Learning rate
    net.run(balancingtime, report='text')
    toc=time.time()-tic
    print('run %.1f seconds.\n'%toc)
    net.store('balanced')

    # --------------------------------------------------
    # Test pulse  without plasticity
    # --------------------------------------------------

    print('Test with stimulus pulse')
    tic=time.time()
    eta = 0          # Learning rate
    net.run(testtime, report='text')
    toc=time.time()-tic
    print('run %.1f seconds.\n'%toc)

    # --------------------------------------------------
    # Run with plasticity
    # --------------------------------------------------


    print('simulating the network')
    tic=time.time()
    eta = 0.0005          # Learning rate
    eta_rec = eta
    net.run(runtime, report='text')
    toc=time.time()-tic
    print('run %.1f seconds.\n'%toc)
    net.store('postrun')

    # --------------------------------------------------
    # Test pulse  without plasticity
    # --------------------------------------------------

    print('Test with stimulus pulse')
    tic=time.time()
    eta = 0          # Learning rate
    net.run(testtime, report='text')
    toc=time.time()-tic
    print('run %.1f seconds.\n'%toc)


    # ###########################################
    # RESULTS
    # ###########################################

    i, t = sm.it
    i_ec, t_ec = sm2.it

    if EC_input_on:
        np.savez_compressed(datetime.datetime.now().strftime((fnamebase+"_RUNTIME_{}s_-_%d-%b-%Y_-_%H_%M_%S_-_pff{}_prc{}_tauE{}_tauSTDP{}_eta{}_DATA_RUN.npz")).format(int(runtime/second),pff,prc,tau_E,tau_stdp, eta_rec), i=i, t=t, i_ec=i_ec, t_ec=t_ec)
    else:
        np.savez_compressed(datetime.datetime.now().strftime((fnamebase+"_RUNTIME_{}s_-_%d-%b-%Y_-_%H_%M_%S_-_pff{}_prc{}_tauE{}_tauSTDP{}_eta{}_DATA_CONTROL.npz")).format(int(runtime/second),pff,prc,tau_E,tau_stdp, eta_rec), i=i, t=t, i_ec=i_ec, t_ec=t_ec)
    print('Mean rate of EC inputs: 1/{:.5f}'.format(np.mean(np.fromiter((np.mean(np.diff(t_ec[np.where(i_ec==x)[0]])) for x in range(Nassembly)),dtype=float))))

    # # ###########################################
    # # Make plots
    # # ###########################################

    # --------------------------------------------------
    # Plot of sequence activation during test pulses

    if EC_input_on:
    
        plt.figure()
        plt.subplot(412)
        plt.plot(t/ms, i, 'k.', ms=0.25)
        plt.xlabel("time (ms)")
        plt.yticks([])
        plt.title("Pre-Run")
        plt.xlim((balancingtime)/ms-100, (balancingtime + testtime)/ms)
        plt.ylim(0,NEin1assem*Nassembly)

        plt.subplot(411)
        plt.plot(t/ms, i, 'k.', ms=0.25)
        plt.xlabel("time (ms)")
        plt.yticks([])
        plt.title("Post-Run")
        plt.xlim((balancingtime+runtime+testtime)/ms-100, (balancingtime + runtime + 2*testtime)/ms)
        plt.ylim(0,NEin1assem*Nassembly)

    else:

        plt.subplot(414)
        plt.plot(t/ms, i, 'k.', ms=0.25)
        plt.xlabel("time (ms)")
        plt.yticks([])
        plt.title("Pre-Run (control)")
        plt.xlim((balancingtime)/ms-100, (balancingtime + testtime)/ms)
        plt.ylim(0,NEin1assem*Nassembly)

        plt.subplot(413)
        plt.plot(t/ms, i, 'k.', ms=0.25)
        plt.xlabel("time (ms)")
        plt.yticks([])
        plt.title("Post-Run (control)")
        plt.xlim((balancingtime+runtime+testtime)/ms-100, (balancingtime + runtime + 2*testtime)/ms)
        plt.ylim(0,NEin1assem*Nassembly)

#plt.show()


plt.rc('figure', titlesize=6)
plt.rc('xtick', labelsize=4)
plt.rc('ytick', labelsize=4)
plt.rc('axes', titlesize=4)
plt.rc('axes', labelsize=4)
plt.savefig(datetime.datetime.now().strftime((fnamebase+"_RUNTIME_{}s_-_%d-%b-%Y_-_%H_%M_%S_-_pff{}_prc{}_tauE{}_tauSTDP{}_eta{}.pdf")).format(int(runtime/second),pff,prc,tau_E,tau_stdp, eta_rec))
# --------------------------------------------------












