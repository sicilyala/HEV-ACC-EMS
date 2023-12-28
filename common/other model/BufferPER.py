from collections import namedtuple, deque
import random
import numpy as np

PBAR_MULT = 5

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    
    def __init__(self, config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            self.batch_size (int): size of each training batch
        """
        self.buffer_size = config['buffer_size']
        self.memory = deque(maxlen=self.buffer_size)  # internal memory (deque)
        self.action_size = config['action_size']
        self.batch_size = config['batch_size']
        self.alpha = config['alpha']
        self.emin = config['emin']
        self.seed = random.seed(config['seed'])
        self.num_agents = config['num_agents']
        # self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward", "next_state", "done", "deltaQ",
                                                  "deltaQ_alpha"])
        self.sum_deltaQ = np.zeros(
            self.num_agents)  # agents have their on deltaQs so there are no overlaps in priorities
        self.sum_deltaQA = np.zeros(self.num_agents)
        self.mean_deltaQ = np.zeros(self.num_agents)
        self.add_deltaQ_mult = 3  # Multplier to mean_deltaQ with which to add new experiences lacking Qs
        self.initial_deltaQ = 0.005  # deltaQ to use for first ekements added
        self.first_batch = False
        # some debug flags
        self.debug_update = False
    
    def add(self, states, actions, rewards, next_states, dones):
        """Add a new experience to memory."""
        # print('Adding experiences to memory : ')
        # print('State shape : ',states.shape)
        # print(type(states))
        new_experience = []
        deltaQ = np.zeros(2)
        deltaQ_alpha = np.zeros(2)
        # new_experience_set = [self.experience() for e in np.arange(states.shape[0])]
        for a in range(self.num_agents):
            if self.first_batch:
                deltaQ[a] = self.mean_deltaQ[a]*self.add_deltaQ_mult
            else:
                deltaQ[a] = self.initial_deltaQ
            deltaQ_alpha[a] = deltaQ[a]**self.alpha
        
        # print('deltaQ : ',deltaQ)
        if len(self) >= self.buffer_size:
            # print('Buffer Full - wait a second: Removing one element')
            # if full pop from left and correct sum
            epop = self.memory.popleft()
            for a in range(self.num_agents):
                self.sum_deltaQ[a] -= epop[a].deltaQ
                self.sum_deltaQA[a] -= epop[a].deltaQ_alpha
        
        for a in range(self.num_agents):  # a as num_agents
            e = self.experience(states[a, :], actions[a, :], rewards[a], next_states[a, :], dones[a], deltaQ[a],
                                deltaQ_alpha[a])
            new_experience.append(e)
        # print('New Experience : ',new_experience)
        self.memory.append(new_experience)
        for a in range(self.num_agents):
            self.sum_deltaQ[a] += deltaQ[a]
            self.sum_deltaQA[a] += deltaQ_alpha[a]
            self.mean_deltaQ[a] = self.sum_deltaQ[a]/len(self)
        
        # print('Sum {} and Mean {}'.format(self.sum_deltaQ, self.mean_deltaQ))
    
    def p_from_qa(self, qa, agent_id):
        return qa/self.sum_deltaQA[agent_id]
    
    def ReCalcSumQ(self):
        for a in range(self.num_agents):
            self.sum_deltaQ[a] = sum([e[a].deltaQ for e in self.memory]).item()
            self.sum_deltaQA[a] = sum([e[a].deltaQ_alpha for e in self.memory]).item()
    
    def updatedeltaQ(self, NewQ, exp_indices, agent_id):
        
        if self.debug_update:
            print('----PER updates following indicies and New Qs (agent : {})-----'.format(agent_id))
            OldQ = []
            # print(np.transpose(NewQ))
            # print(exp_indices)
        
        for ei, q in zip(exp_indices, NewQ):
            q = q.item()
            if self.debug_update:
                # print('Updating index : {} with new Q : {:6.4f} while previous value is {:6.4f}'.format(ei, q,self.memory[ei][agent_id].deltaQ))
                OldQ.append(self.memory[ei][agent_id].deltaQ)
            self.sum_deltaQ[agent_id] += (q-abs(self.memory[ei][agent_id].deltaQ))
            self.sum_deltaQA[agent_id] += (q**self.alpha-abs(self.memory[ei][agent_id].deltaQ)**self.alpha)
            self.memory[ei][agent_id] = self.memory[ei][agent_id]._replace(deltaQ=q)
            self.memory[ei][agent_id] = self.memory[ei][agent_id]._replace(deltaQ_alpha=q**self.alpha)
        
        if self.debug_update:
            print('MinNewQ : {:8.6f} MaxNewQ : {:8.6f} and mean NewQ = {:8.6f} and std NewQ = {:8.6f}'.format(
                np.min(NewQ), np.max(NewQ), np.mean(NewQ), np.std(NewQ)))
            print('MinOldQ : {:8.6f} MaxOldQ : {:8.6f} and mean OldQ = {:8.6f} and std OldQ = {:8.6f}'.format(min(OldQ),
                                                                                                              max(OldQ),
                                                                                                              np.mean(
                                                                                                                  OldQ),
                                                                                                              np.std(
                                                                                                                  OldQ)))
    
    def drawsample(self, numdraws, agent_id):
        """ Draw prioritized sample according to probabilities by deltaQs stored 
        
        Params
        ======
            numdraws (int): number of experiences to sample """
        
        Sampling_Debug_DS = False
        experiences = []
        drawn_indices = []
        probs = []
        # Keeping track of number accepted and rejected
        numacc = 0
        numre = 0
        n = len(self)
        # Sampling with acceptance until numdraws draws are accepted
        for i in np.arange(numdraws):
            accepted = False
            while not accepted:
                curdraw = np.random.randint(0, n)
                
                # print('Curdraw : ',curdraw,' while ',drawn_indices,' have been drawn.')
                if curdraw not in drawn_indices:
                    e = self.memory[curdraw]
                    p = self.p_from_qa(e[agent_id].deltaQ_alpha, agent_id).item()
                    # accept with prob=1 if p > 2*pbar or with prob = p/(2*pbar), PBAR_MULT is being 2 in this case
                    accepted = (p/PBAR_MULT*n > np.random.uniform())
                    if accepted:
                        numacc += 1
                        drawn_indices.append(curdraw)
                        experiences.append(e)
                        probs.append(p)
                        # print('Accepted {} on {}'.format(curdraw,i))
                    else:
                        numre += 1
                        # print('Rejected {} on {}'.format(curdraw,i))
        
        if Sampling_Debug_DS:
            print('Final Sample : (acc : {} -- rej : {} )'.format(numacc, numre))
            # print(drawn_indices)
            # np.set_printoptions(precision=4)
            # print(np.array(probs).transpose())
        
        return experiences, drawn_indices, np.asarray(probs)
    
    def register_batch(self):
        if not self.first_batch:
            self.first_batch = True
    
    def ERsample(self):
        """Randomly sample a batch of experiences from memory."""
        self.register_batch()
        exp_indices = random.sample(range(len(self.memory)), k=self.batch_size)
        experiences = [self.memory[i] for i in exp_indices]
        probs = np.ones(self.batch_size)/len(self)
        return experiences, exp_indices, probs
    
    def sample(self):
        """Prioritized Randomly sample a batch of experiences from memory."""
        self.register_batch()
        experiences = [None]*2
        exp_indices = [None]*2
        probs = [None]*2
        for a in range(self.num_agents):
            experiences[a], exp_indices[a], probs[a] = self.drawsample(self.batch_size, a)
        # experiences = random.sample(self.memory, k=self.batch_size)
        return experiences, exp_indices, probs
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
    def printdeltaQs(self):
        ShowQandP = False
        np.set_printoptions(precision=4)
        if ShowQandP: print('Showing Qs')
        probs = np.zeros(len(self))
        Qs = np.zeros(len(self))
        
        for e, i in zip(self.memory, np.arange(len(self))):
            Qs[i] = e.deltaQ
        
        if ShowQandP: print(Qs)
        if ShowQandP: print('Associated Probs : ')
        probs = np.zeros(len(self))
        for e, i in zip(self.memory, np.arange(len(self))):
            prob = self.p_from_q(e.deltaQ)
            probs[i] = prob
        
        if ShowQandP: print(probs)
        print('MinP : {:8.6f} MaxP : {:8.6f} and Pbar = {:8.6f} should be 1/n = {:8.6f} with n = {} '.format(min(probs),
                                                                                                             max(probs),
                                                                                                             np.mean(
                                                                                                                 probs),
                                                                                                             1/len(
                                                                                                                 self),
                                                                                                             len(self)))
        print('Max/pbar = {:6.4f}'.format(max(probs)/np.mean(probs)))
        print('MinQ : {:8.6f} MaxQ : {:8.6f} and Qbar = {:8.6f}'.format(min(Qs), max(Qs), np.mean(Qs)))
        print('MinabsQ : {:8.6f} MaxabsQ : {:8.6f} and absQbar = {:8.6f}'.format(min(abs(Qs)), max(abs(Qs)),
                                                                                 np.mean(abs(Qs))))
        
        return
    
    def mem_print_summary(self):
        
        print('----------------------Replay Buffer Summary---------------------------------')
        print('Exponent for weights : ', self.alpha)
        print('Number of Elements in Buffer : ', len(self))
        
        AllQ = np.zeros((len(self), self.num_agents))
        AllQA = np.zeros((len(self), self.num_agents))
        AllP = np.zeros((len(self), self.num_agents))
        for a in range(self.num_agents):
            for e, i in zip(self.memory, np.arange(len(self))):
                AllQ[i, a] = e[a].deltaQ
                AllQA[i, a] = e[a].deltaQ
            WeightedSumQ = sum(abs(AllQ[:, a]))
            WeightedSumQA = sum(abs(AllQ[:, a])**self.alpha)
            for e, i in zip(self.memory, np.arange(len(self))):
                AllP[i, a] = self.p_from_qa(e[a].deltaQ_alpha, a)
        
        for a in range(self.num_agents):
            # print(AllQ)
            CurQ = AllQ[:, a]
            CurP = AllP[:, a]
            print('Agent {} : '.format(a))
            print('MinabsQ : {:11.9f} MaxabsQ : {:11.9f} and meanQ = {:11.9f} while mean(abs(Q)) = {:11.9f}'.format(
                min(abs(CurQ)), max(abs(CurQ)), np.mean(CurQ), np.mean(abs(CurQ))))
            print('With weighted sum from AllQ being : {} and stored one is : {}'.format(WeightedSumQ,
                                                                                         self.sum_deltaQ[a]))
            print('With weighted sum from AllQA being : {} and stored one is : {}'.format(WeightedSumQA,
                                                                                          self.sum_deltaQA[a]))
            # print(AllQ)
            # print(abs(AllQ)**self.alpha)
            
            print('MinabsP : {:11.9f} MaxabsP : {:11.9f} and meanP = {:11.9f}'.format(min(abs(CurP)), max(abs(CurP)),
                                                                                      np.mean(CurP)))
            print('With sum from AllP being : {} hopefully being 1.'.format(sum(CurP)))
            print('Max/pbar = {:6.4f}'.format(max(CurP)/np.mean(CurP)))
        
        return AllQ, AllP