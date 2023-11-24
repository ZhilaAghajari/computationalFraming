import numpy as np
from tqdm import tqdm
from word import Word



class ThetaRoleModel:
    def __init__(self, corpus, originaltext, doc_relns, doc_arg2, vocab_relns, vocab_args, id2word, reln2id, arg2id,  n_iters, K, T, D, V, R, A2, alpha, eta, etaprime, gamma, lam, omega):
        self.corpus = corpus
        self.doc_relns = doc_relns
        self.doc_arg2 = doc_arg2
        self.vocab_relns = vocab_relns
        self.vocab_args = vocab_args
        self.id2word = id2word
        self.reln2id = reln2id
        self.arg2id = arg2id
        self.corpus_hat = None # corpus in the form of Word objects
        self.originaltext = originaltext
       
        self.n_iters = n_iters

        # n topics, n theta roles, n documents, n words, n relations: https://universaldependencies.org/u/dep/
        #n arg2
        self.K = K
        self.T = T
        self.D = D
        self.V = V
        self.R = R
        self.A2 = A2

        # parameters for dirichlet distribution(s)
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        #self.omega = omega
        self.etaprime = etaprime

        # matrices from plate diagram
        self.theta = None # documents x topic matrix 
        self.beta = None # topics x vocabulary
        
        self.phi = None # documents x theta roles matrix
        self.zeta = None # theta roles x grammatical relationships
        self.betaprime = None
        
        
        
        # counts for gibbs samplins
        self.n_d_k = None #count of occurance of topic k in document d
        self.n_k_w = None #counts of word w in the topic k
        self.n_t_w = None #counts of word w in the theta role t
        self.n_t_reln = None #counts of reln in the theta role t
        self.n_t_arg2 = None #counts of arg2 in the theta role t
        self.n_w_reln = None
        self.n_w_arg2 = None
        self.n_k = None #count of topics
        self.n_d_t = None #counts of theta role t for document d
        self.n_t_k = None #count of topic k for theta role t 
        self.n_t = None #AGAIN what is this????
        
        #pribability of theta role. it is for testing. otherwise it wouldn't be a class variable.
        self.p_t_reln_w_z_d = None


    def initialize_variables(self):
        '''
        Initialize Word objects to store a random topic and theta role for gibbs sampling
        '''
        # initalize counts for gibbs sampling
        self.n_d_k = np.zeros((self.D, self.K)) # count of occurance of topic k in document d
        self.n_k_w = np.zeros((self.K, self.V)) # counts of word w in the topic k
        self.n_t_w = np.zeros((self.T, self.V)) # counts of word w in the theta role t
        self.n_k = np.zeros(self.K) #count of topics
        self.n_d_t = np.zeros((self.D, self.T)) #counts of theta role t for document d
        
        self.n_t_k = np.zeros((self.T, self.K)) #count of topic k for theta role t
        
        self.n_t_reln = np.zeros((self.T, self.R)) #count of grammatical relationships reln for theta role t?
        self.n_t_arg2 = np.zeros((self.T, self.A2)) 
        self.n_t = np.zeros(self.T) #is it number of topics for theta role t??? what is it? and when is it used?
        
        self.n_w_arg2 = np.zeros((self.V, self.A2))
        self.n_w_reln = np.zeros((self.V, self.R))

        self.corpus_hat = []
        for d, doc in enumerate(self.corpus):
            doc_hat = []
            for i, word_id in enumerate(doc):
                z = np.random.randint(self.K) # random topic
                y = np.random.randint(self.T) # random theta role 
                y1 = np.random.randint(self.T)
                y2 = np.random.randint(self.T)
                
                # obtain relns for word i in current document
                relns = self.doc_relns[d][i]
                arg2s = self.doc_arg2[d][i]
                if not relns:
                    continue
                
                # initalize counts
                self.n_d_k[d, z] += 1
                self.n_k_w[z, word_id] += 1 #add to this topic term frequency
                self.n_k[z] += 1 #number of words with this topic z 
                
                
                #for initalization,  we can share a y between these two? otherwise we should do y --> y1, and y -->y2
                for reln in relns:
                    #self.n_t_reln[y1, self.reln2id[reln]] += 1
                    self.n_t_reln[y, self.reln2id[reln]]+=1
                    self.n_w_reln[word_id, self.reln2id[reln]]+=1
                for arg in arg2s: 
                    #self.n_t_arg2[y2, self.arg2id[arg]]+=1
                    self.n_t_arg2[y, self.arg2id[arg]]+=1
                    self.n_w_arg2[word_id, self.arg2id[arg]]+=1
                    
                #self.n_d_t[d, y] += 1 #we don't have this because our theta role is outside the document 
                self.n_t_k[y, z] += 1 #count of topic k for theta role t
                self.n_t[y] += 1 #is it total count of "word" within theta role y. or is it totall number of time theta role y is chosen????the latter
                
                self.n_t_w[y, word_id]+=1 #count of words for theta role y; 
                #wil we have it even though there is no link from y1 and y2 to observed variable word in above plate? PERHAPS NOT!
                
                word = Word(content=self.id2word[word_id], idx=word_id, z=z, y=y, y1=y1, y2=y2, relns=relns, arg2s=arg2s)
                doc_hat.append(word) #each doc_hat is a list of word object that appear in thids document

            self.corpus_hat.append(doc_hat)

    def fit(self):
        '''
        Run gibbs sampling for n iterations to assign theta role and topic to each word
        '''
        for i in tqdm(range(self.n_iters)):
            for d, doc in enumerate(self.corpus_hat):
                N = len(doc)

                for word in doc:
                    z, y, y1, y2, idx, relns, arg2s = word.z, word.y, word.y1, word.y2, word.idx, word.relns, word.arg2s
                
                    # subtract count for current topic and theta role;  
                    self.n_d_k[d, z] -= 1
                    self.n_k_w[z, idx] -= 1 
                    self.n_t_w[y, idx] -=1 # XX we probably won't use this? we will have one for arg2 and reln though
                    self.n_k[z] -= 1
                    for reln in relns:
                        self.n_t_reln[y, self.reln2id[reln]] -= 1
                        #self.n_t_reln[y1, self.reln2id[reln]] -= 1
                        self.n_w_reln[idx, self.reln2id[reln]]-=1
                    for arg in arg2s:
                        #self.n_t_arg2[y2, self.arg2id[arg]] -= 1
                        self.n_t_arg2[y, self.arg2id[arg]] -= 1
                        self.n_w_arg2[idx, self.arg2id[arg]]-=1
                        #self.n_arg[arg]-=1 #this is observed no need to add it in fit.
                        #self.n_t_k[y1, z] -= 1
                    self.n_t_k[y, z] -= 1 
                    self.n_t[y] -= 1

                    # calculate conditional probability distribution for topic selection
                    #######################1
                    # p(k|w,d) =~ p(w | k)* p(k | d) 
                    p_k = np.zeros(self.K)
                    p_w_k = (self.n_k_w[:, idx] + self.eta) / (self.n_k[:] + self.eta * self.V)
                    p_k_d = (self.n_d_k[d, :] + self.alpha) / (N + self.K * self.alpha)
                    p_k = p_k_d * p_w_k
                    #######################1
#                   
                    z = np.random.choice(self.K, p=p_k / sum(p_k))
                    word.z = z
                    
                    # update counts for the selected topic
                    self.n_d_k[d, z] += 1
                    self.n_k_w[z, idx] += 1
                    
                    self.n_k[z] += 1

                    # find probability of theta roles,given current grammatical relations of this word
                    # and the seclected topic z
                    
                    p_t_reln = np.zeros((self.T, self.R))
                    p_t_arg2 = np.zeros((self.T, self.A2))
                    #######################2

                    #p_t_reln_k :p(y1|reln, phi_z) ~ p(reln|y1)*p(y1|phi_z)
                    #p(y1|phi_z)
                    p_z_t = (self.n_t_k[:, z]) \
                                 / (np.sum(self.n_t_k[:,z]))
                    p_t_reln_k = np.zeros(self.T)
                    p_t_arg2_k = np.zeros(self.T)
                    #p(reln|y1)
                    for reln in relns:
                        #vector size of t for y1s
                        p_t_reln[:, self.reln2id[reln]] = (self.n_t_reln[:, self.reln2id[reln]] + self.lam) \
                                     / (np.sum(self.n_t_reln[:, :],axis=1) + self.lam * self.R) 
                    p_t_reln_k = np.sum(p_t_reln, axis =1)*p_z_t #Is p_z_t scaler for each theta role t? 
                     
                        
                    for arg in arg2s:
                        p_t_arg2[:, self.arg2id[arg]] = (self.n_t_arg2[:, self.arg2id[arg]] + self.etaprime) \
                                     / (np.sum(self.n_t_arg2[:, :],axis=1) + self.etaprime * self.A2)
                    p_t_arg2_k = np.sum(p_t_arg2, axis = 1)*p_z_t
                    
                    
                    #p_t1 = np.sum(p_t_reln, axis=0)
                    p_t1 = p_t_reln_k / np.sum(p_t_reln_k)
                    #print('*********** , ', p_t1)
                    y1 = np.random.choice(self.T, p=p_t1 )
                    word.y1 = y1
            
                    #p_t2 = np.sum(p_t_arg2_k, axis=0)
                    
                    p_t2 = p_t_arg2_k/np.sum(p_t_arg2_k)
                    #print('ARE U NEGATIVE? ', p_t2)
                    y2 = np.random.choice(self.T, p=p_t2)
                    word.y2 = y2

                    if (p_t1[y1])>=(p_t2[y2]):
                        word.y = y1
                    else:
                        word.y = y2
                    
                    probs = np.array([p_t1[y1], p_t2[y2]])
                    probs = probs/np.sum(probs)
                    y = np.random.choice([y1, y2], p = probs)
                    word.y = y
                    
                    # update counts for selected theta role
                    for reln in relns:
                        #self.n_t_reln[y1, self.reln2id[reln]] += 1
                        self.n_t_reln[y, self.reln2id[reln]] += 1
                        self.n_w_reln[idx, self.reln2id[reln]]+=1
                    for arg in arg2s:
                        #self.n_t_arg2[y2, self.arg2id[arg]] += 1
                        self.n_t_arg2[y, self.arg2id[arg]] += 1
                        self.n_w_arg2[idx, self.arg2id[arg]]+=1
#                         self.n_w_arg[idx,arg]+=1
                        
                    #self.n_d_t[d, y] += 1 #X
                    #???????? ah maybe there is no actual y? and maybe we update self.n_t_k[y, z] for y1 and y2? not sure here..
                    self.n_t_k[word.y, z] += 1 
                    self.n_t[word.y] += 1 #number of words assigned to this theta role y       
                    #self.n_t_k[y1, z] += 1 
                    #self.n_t[y1] += 1
                    #self.n_t_k[y2, z] += 1 
                    #self.n_t[y2] += 1
                    self.n_t_w[y, idx] +=1 #number of times this word and theta y occured together..
                    
    
    def compute_phi(self):
        '''
        Compute the phi matrix from the plate diagram 
        '''
        self.phi = np.zeros((self.K, self.T))
        for k in range(self.K):
            for t in range(self.T):
                #@@@@CHECK THIS ONE
                self.phi[k, t] = (self.n_t_k[t, k] + self.gamma) / (np.sum(self.n_t_k[:, k]) + self.T * self.gamma)  
        return self.phi
    

    def compute_theta(self):
        '''
        Compute the theta matrix from the plate diagram
        '''
        self.theta = np.zeros((self.D, self.K))
        for d in range(self.D):
            _N = len(self.corpus[d])
            self.theta[d] = (self.n_d_k[d] + self.alpha) / (_N + self.K * self.alpha)
        return self.theta

    
          

    def compute_beta(self):
        '''
        Compute the beta matrix from the plate diagram
        '''
        self.beta = np.zeros((self.K, self.V))
        for k in range(self.K):
            self.beta[k] = (self.n_k_w[k] + self.eta) / (self.n_k[k] + self.V * self.eta)
        return self.beta
    
    #not sure if we use it but we may need a beta prime for theta role and word probability... the new arrow we added ... 
    def compute_betaprime(self):
        '''
        Compute the beta matrix from the plate diagram
        '''
        self.betaprime = np.zeros((self.T, self.A2))
        for t in range(self.T):
            self.betaprime[t] = (self.n_t_arg2[t] + self.etaprime) / (np.sum(self.n_t_arg2[t]) + self.A2 * self.etaprime)
            #Instead of w, we should have arg2 ? 
            #self.betaprime[t] = (self.n_t_w[t] + self.etaprime) / (self.n_t[t] + self.V * self.etaprime)
        return self.betaprime

    #Verify this one!!!
    def compute_zeta(self):
        '''
        Compute the zeta matrix from the plate diagram
        '''
        self.zeta = np.zeros((self.T, self.R))
        for t in range(self.T):
            self.zeta[t] = (self.n_t_reln[t] + self.lam) / (np.sum(self.n_t_reln[t]) + self.R * self.lam) 
            #self.zeta[t] = (self.n_t_reln[t] + self.lam) / (self.n_t[t] + self.R * self.lam) 
        return self.zeta
    

    def print_topics(self):
        '''
        Print topics
        '''
        for k in range(self.K):
            word_ids = np.argsort(self.beta[k])[::-1][:20]
            probs = np.sort(self.beta[k])[::-1][:20]
            top_words = [self.id2word[i] for i in word_ids]
            strings = [f'{prob} * {word}' for prob, word in zip(probs, top_words)]
            print(f"Topic {str(k)}: {', '.join(strings)} \n") 
            
    
    def print_all_results(self):
        '''
        print topics word, and their reln and arg2 distributions
        '''
        top_doc = 10
        tops = 1
        for k in range(self.K):
            word_ids = np.argsort(self.beta[k])[::-1][:20]
            probs = np.sort(self.beta[k])[::-1][:20]
            top_words = [self.id2word[i] for i in word_ids]
            strings = [f'{prob} * {word}' for prob, word in zip(probs, top_words)]
            print(f"Topic {str(k)}: {', '.join(strings)} \n") 

            top_theta_ids = self.top_theta_role_for_topic(k, tops)
            #maybe we should get 3 top theta role instead of one top theta ?
            print(' *** Top theta roles {0} - {1} ** \n'.format(str(top_theta_ids[0]), str(self.phi[k, top_theta_ids[0]])))
            print('Distribution of reln for each words of the topic {0}'.format(k))
            for t in top_theta_ids:
                dic_reln = {reln:0 for reln in self.vocab_relns} 
                results = [(self.vocab_relns[j], self.zeta[t,j]) for j in range(len(self.zeta[t,:]))]
                results = dict(results)
                results = sorted(results.items(), key=lambda x: x[1], reverse=True)
                print(f"* Relns {', '.join([result[0]+': '+str(result[1]) for result in results[0:20]])} \n")
                #they do not sum to 1
        
                #print("Distribution of arg2 for theta role {0}\n".format(str(top_theta_ids[t])))
                dic_arg2 = {arg:0 for arg in self.vocab_args} 
                res_arg2 = [(self.vocab_args[j], self.betaprime[t,j]) for j in range(len(self.betaprime[t,:]))]
                res_arg2 = dict(res_arg2)
                res_arg2 = sorted(res_arg2.items(), key=lambda x: x[1], reverse=True)
                print(f"* Args {', '.join([arg[0]+': '+str(arg[1]) for arg in res_arg2[0:20]])} \n")
            #print('Distribution of arg2 for each words of the topic {0}'.format(k))
            for widx in  word_ids:
                dic_reln = {reln:0 for reln in self.vocab_relns} #
                for t in top_theta_ids:
                    dist1 = self.reln_distribution_for_word_idx(widx, t)
                    for ki,v in dist1.items():
                        dic_reln[ki] += v
                #sort based on probability of reln for all the theta role...
                results_relns = sorted(dic_reln.items(), key=lambda x: x[1], reverse=True)
                print(f"* {str(self.id2word[widx])}: Relns: {', '.join([result[0]+':'+str(result[1]) for result in results_relns[0:3]])} \n")
                
                
                dic_arg2 = {reln:0 for reln in self.vocab_args} #
                for t in top_theta_ids:
                    dist = self.arg2_distribution_for_word_idx(widx, t)
                    for ki,v in dist.items():
                        dic_arg2[ki] += v
                #sort based on probability of args for all the theta role...
                results = sorted(dic_arg2.items(), key=lambda x: x[1], reverse=True)
                print(f"* {str(self.id2word[widx])}: Arg2: {', '.join([result[0]+':'+str(result[1]) for result in results[0:3]])} \n")
                print('-------- -------- -------- -------- -------- --------')

    
    def count_arg(self, arg):
        c=0
        for ls in self.doc_arg2:
            for ls2 in ls:
                c+=ls2.count(arg)
        return c
    
    def arg2_distribution_for_word_idx(self, widx, t):
        #p(arg2|widx, th=t, topic=z) ~ p(w|arg2)*p(arg2|th=t)*p(th=t|topic=z) wherein the last element is constant
        
        res = {}
        for arg in self.vocab_args:
            #p(w|arg2) : p_w_arg
            p_w_arg= self.n_w_arg2[widx, self.arg2id[arg]]/self.count_arg(arg)
            p_arg_theta = self.betaprime[t, self.arg2id[arg]]
            res[arg] = p_w_arg*p_arg_theta
        return res
    
    
    def count_reln(self, reln):
        c=0
        for ls in self.doc_relns:
            for ls2 in ls:
                c+=ls2.count(reln)
        return c
    
    def reln_distribution_for_word_idx(self, widx, t):
        res ={}
        for reln in self.vocab_relns:
            p_w_reln = self.n_w_reln[widx, self.reln2id[reln]]/self.count_reln(reln)
            p_reln_theta = self.zeta[t, self.reln2id[reln]]
            res[reln]=p_w_reln*p_reln_theta
        return res
    
    def top_theta_role_for_topic(self, k, tops): #k is topic id 
        top_theta_indx = np.argsort(self.phi[k, :])[::-1][:tops]
        return top_theta_indx


    #self.betaprime = np.zeros((self.T, self.V))
    def top_theta_role_for_wordidx(self, widx, tops):
        '''
        get the top theta role t for this specic word with id widx
        '''
        top_theta_indx = np.argsort(self.betaprime[:, widx])[::-1][:tops]
        return top_theta_indx
    
    def compute_reln_distribution_for_wordidx_in_top_theta_t(self, widx, thetaroleidx):
        '''
        get the p(reln|word within the top theta role thetat)
        '''
        res = {}
        for i in range(len(self.vocab_relns)):###
            
            p_reln_y = self.zeta[thetaroleidx, self.reln2id[self.vocab_relns[i]]]\
            /np.sum(self.zeta[thetaroleidx])
            
    
            p_t_w = self.betaprime[thetaroleidx,widx]\
            / np.sum(self.betaprime[thetaroleidx])
            p_relni_widx_thetaroleidx = p_reln_y * p_t_w
            res[self.vocab_relns[i]] = p_relni_widx_thetaroleidx

        return res   
   
    

    
    #after gips sampling: we want to know for each word in a given topic the distribution of typed dependies; 
    #it should be in the context of topic and top theta role.. so it shoudl recieve the topic number and theta role number, and word_indx? 
    # p(reln|w_indx, topic_k, theta_t) ~ p(reln, theta_t)p(theta_t, topic_k)p(topic_k, word_indx) 
    def print_topics_word_reln_distribution(self):
        '''
        Print topics with words and each words' reln distribution
        '''
        top_doc = 10
        for k in range(self.K):
            word_ids = np.argsort(self.beta[k])[::-1][:20]
            probs = np.sort(self.beta[k])[::-1][:20]
            top_words = [self.id2word[i] for i in word_ids]
            strings = [f'{prob} * {word}' for prob, word in zip(probs, top_words)]
            print(f"Topic {str(k)}: {', '.join(strings)} \n") 
            #Add a distribution of p(reln|word, topic, top theta role)
            #First, need to know top theta role for this specific topic based on distribution of each theta role for top documents...
            top_documents_indx = np.argsort(self.theta[:, k])[::-1][:top_doc]
            top_theta_ids = self.top_theta_role_for_topic(k, top_documents_indx)
            print(' top theta ids : ', top_theta_ids)
            for widx in word_ids:
                dic_reln = {reln:0 for reln in self.vocab_relns} #
                for t in top_theta_ids:
                    dist = self.compute_reln_distribution_for_topic_words(widx, k, t)
                    #merge dist for all the theta role for this topic..sum dist or average dist
                    for ki,v in dist.items():
                        dic_reln[ki] += v
                #sort based on probability of reln for all the theta role...
                results = sorted(dic_reln.items(), key=lambda x: x[1], reverse=True)
                print('-------- -------- -------- -------- -------- --------')
                print(f"* {str(self.id2word[widx])}: Relns: {', '.join([result[0]+':'+str(result[1]) for result in results[0:3]])} \n")
            
    
    def cfs(self, w):
        c = 0
        for sublist in self.corpus:
            c+= sublist.count(w)
        return c
        
    
    #they should be chosen for words within each topic... 
    def compute_reln_distribution_for_topic_words(self, widx, topicidx, thetaroleidx):
        res = {}
        #len(set(reln2id)) is 98!
        #p_reln = [0]*98
        #get the top reln for this theta role
        reln_ids = np.argsort(self.zeta[thetaroleidx])[::-1][:]
        top_relns = [self.vocab_relns[i] for i in reln_ids] #name of these reln ids
        #compute p(reln|w_indx, topic_k, theta_t)
#         for i in reln_ids:
        for i in range(len(self.vocab_relns)):###
            #p(reln, theta_t)
            p_reln_y =(self.n_t_reln[thetaroleidx, self.reln2id[self.vocab_relns[i]]] + self.lam) \
            / (self.n_t[thetaroleidx] + self.lam * self.R)
            #p(theta_y, topic_k)
            p_y_z = (self.n_t_k.T[topicidx, thetaroleidx] + self.gamma) \
            / (self.n_k[topicidx] + self.gamma * self.K)
            #p(topic_k, word_indx)
            p_z_w = (self.n_k_w.T[widx, topicidx] + self.eta) / (self.cfs(widx) + self.eta * self.V)
            p_reln_w_z_y = p_reln_y * p_y_z * p_z_w
            #need to construct reln:prob here and return it as a string of reln distributions... maybe a tuple or dic so we can sort it based on probability scores?
            res[self.vocab_relns[i]] = p_reln_w_z_y
            #res[i] = p_reln_w_z_y
        #sort this dicionary based on values and return the tuple of key values ? 
#         results = sorted(res.items(), key=lambda x: x[1], reverse=True)
        return res




    
    
    # p(reln|w, topic k , theta role t) = p(reln|y)p(y|z)p(z|w)
    def compute_reln_w(self, k, t, reln, w):
        p_reln_y = (self.n_t_reln[t, self.reln2id[reln]] + self.lam) \
            / (self.n_t[t] + self.lam * self.R)
        p_y_z = (self.n_t_k.T[k, t] + self.gamma) \
            / (self.n_k[k] + self.gamma * self.K)
        #self.cfs is count frequency for wordds but it is not defined...
        p_z_w = (self.n_k_w.T[w, k] + self.eta) / (self.cfs(w) + self.eta * self.V)
        p_reln_w = p_reln_y * p_y_z * p_z_w

        return p_reln_w
    
        
# add the top documents as well

    def print_theta_roles_relns(self):
        '''
        Print theta roles
        '''
        for t in range(self.T):
            reln_ids = np.argsort(self.zeta[t])[::-1][:20]
            probs = np.sort(self.zeta[t])[::-1][:20]
            top_relns = [self.vocab_relns[i] for i in reln_ids]
            strings = [f'{prob} * {reln}' for prob, reln in zip(probs, top_relns)]
            print(f"Theta Role \n {str(t)}: {', '.join(strings)}\n")
    
    def print_theta_roles_args(self):
        '''
        Print theta roles
        '''
        
        for t in range(self.T):
            args_ids = np.argsort(self.betaprime[t])[::-1][:20]
            probs = np.sort(self.betaprime[t])[::-1][:20]
            top_args = [self.vocab_args[i] for i in args_ids]
            strings = [f'{prob} * {arg}' for prob, arg in zip(probs, top_args)]
            print(f"Theta Role \n {str(t)}: {', '.join(strings)}\n")

    def print_top_of_docs(self):
        '''
        Print top topics and theta roles for each document
        in this model, documents do not have theta 
        '''
        pass
            
    def print_top_documents_topic(self, top_doc=10):
        '''
        print top documents for each topic
        '''
        
        for k in range(self.K):
            top_documents_indx = np.argsort(self.theta[:, k])[::-1][:top_doc]
            top_thetas = 10
            print('topic {0} \n'.format(k))
            try:
                for idx in top_documents_indx:
                    print("doc {0} - {1} - {2} \n ".format(idx, self.theta[idx, k], self.originaltext[str(idx)]))
                print('------------------------------------------------------------------------')
                #now get the top theta role for this specific document...
                #COMMENTED THE REST SINCE THE MODEL CHANGED>.
#                 top_thetas_for_doc_indx = np.argsort(self.phi[idx])[::-1][:top_thetas]
#                 try:
#                     for t in top_thetas_for_doc_indx:
#                         print("theta {0} - {1} \n ".format(t, self.phi[idx, t]))                     
#                 except:
#                     print(top_documents_indx)
            except:
                return top_documents_indx
    
    
    ###### NEXT, I need to calculate given a document what is the probability of each topic and each theta role for that document.
    def print_top_topics_doc(self, top_topics=20):
        '''
        for each document, it prints the top topic
        '''
        for idx in range(self.D):
            top_topics_indx = np.argsort(self.theta[idx, :][::-1][:top_topics])
            print('document {0} \n'.format(idx))
            try:
                for k in top_topics_indx:
                    print("topic {0} - {1} \n ".format(k, self.theta[idx, k]))
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
            except:
                return top_topics_indx
        
            
    def print_top_thetas_doc(self, top_thetas=20):
        '''
        print the top theta role for each document
        in the new model, documents do not have theta role..
        '''
        pass
#         for idx in range(self.D):
#             top_thetas_indx = np.argsort(self.phi[idx, :][::-1][:top_thetas])
#             print('document {0} \n'.format(idx))
#             try:
#                 for t in top_thetas_indx:
#                     print("theta {0} - {1} \n ".format(t, self.phi[idx, t]))
#                 print('------------------------------------------------------------------------')
#                 print('------------------------------------------------------------------------')
#             except:
#                 return top_thetas_indx
    
    
    
#     def print_top_documents_theta(self, top_doc=20):
#         for t in range(self.T):
#             top_documents_indx = np.argsort(self.phi[:, t])[::-1][:top_doc]
#             print('Theta number {0} \n'.format(t))
# #             print('index of top documents: ', top_documents_indx)
#             try:
#                 for idx in top_documents_indx:
#                     print("doc {0} - {1} - {2} \n ".format(idx, self.phi[idx, t], self.originaltext[str(idx)]))
#                 print('------------------------------------------------------------------------')
#             except:
#                 return top_documents_indx
    
    
#     def print_top_topics_doc_theta(self, top_topics=20,top_thetas=10, top_doc=20):
#         #get the top topics' first?
#         for k in range(self.K):
#             top_documents_indx = np.argsort(self.theta[:, k])[::-1][:top_doc]
#             try:
#                 for idx in top_documents_indx:
#                     print("doc {0} \n ".format(idx))
#                     print("doc {0} - {1} - {2} \n ".format(idx, self.theta[idx, k], self.originaltext[str(idx)]))
                    
#                     #now get the top theta role for this specific document...
#                     top_thetas_for_doc_indx = np.argsort(self.phi[idx])[::-1][:top_thetas]
#                     try:
#                         for t in top_thetas_for_doc_indx:
#                             print("theta {0} - {1} \n ".format(t, self.phi[idx, t]))
#                     except:
#                         return top_documents_indx
#             except:
#                 return top_documents_indx
                    
    
    def compute_matrices(self):
        _theta = self.compute_theta()
        _phi = self.compute_phi()
        _beta = self.compute_beta()
        _betaprime = self.compute_betaprime()
        _zeta = self.compute_zeta()

        return _theta, _phi, _beta, _betaprime, _zeta


    def print_all(self):
        print('****************** TOPICS TOP TERMS, and their GRAMMATICAL RELN distributions ****************** \n ')
        self.print_all_results()
        print('****************** TOPICS Top Terms  ****************** \n')
        self.print_topics()
        print('****************** TOPICS Top Documents ****************** \n')
        self.print_top_documents_topic()
        print('****************** ****************** ****************** ******************')
        print('++++++++++++++++++ THETA ROLES / Relns ++++++++++++++++++ \n')
        self.print_theta_roles_relns()
        print('++++++++++++++++++ THETA ROLES / Args ++++++++++++++++++ \n')
        self.print_theta_roles_args()
#         self.print_top_documents_theta() #not sure if it is important at all..
        

            
                
            
        
            
          