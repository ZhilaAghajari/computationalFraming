import numpy as np
from tqdm import tqdm
from word import Word

class ThetaRoleModel:
    def __init__(self, corpus, originaltext, doc_relns, vocab_relns, id2word, reln2id, n_iters, K, T, D, V, R, alpha, eta, etaprime, gamma, lam, omega):
        self.corpus = corpus
        self.doc_relns = doc_relns
        self.vocab_relns = vocab_relns
        self.id2word = id2word
        self.reln2id = reln2id
        self.corpus_hat = None # corpus in the form of Word objects
        self.originaltext = originaltext
       
        self.n_iters = n_iters

        # n topics, n theta roles, n documents, n words, n relations: https://universaldependencies.org/u/dep/
        self.K = K
        self.T = T
        self.D = D
        self.V = V
        self.R = R

        # parameters for dirichlet distribution(s)
        self.alpha = alpha
        self.eta = eta
        self.gamma = gamma
        self.lam = lam
        self.omega = omega
        self.etaprime = etaprime

        # matrices from plate diagram
        self.theta = None # documents x topic matrix 
        self.phi = None # documents x theta roles matrix
        self.beta = None # topics x vocabulary
        self.zeta = None # theta roles x grammatical relationships
        self.betaprime = None
        self.psi = None
        
        
        # counts for gibbs samplins
        self.n_d_k = None
        self.n_k_w = None
        self.n_t_w = None
        self.n_k = None #counts of words for each topic
        self.n_d_t = None
        self.n_t_k = None
        self.n_t_reln = None
        self.n_t = None
        
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
        self.n_t = np.zeros(self.T) #is it number of topics for theta role t??? what is it? and when is it used? 

        self.corpus_hat = []
        for d, doc in enumerate(self.corpus):
            doc_hat = []
            for i, word_id in enumerate(doc):
                z = np.random.randint(self.K) # random topic
                y = np.random.randint(self.T) # random theta role

                # obtain relns for word i in current document
                relns = self.doc_relns[d][i]
                if not relns:
                    continue
                
                # initalize counts
                self.n_d_k[d, z] += 1
                self.n_k_w[z, word_id] += 1 #add to this topic term frequency
                self.n_k[z] += 1 #number of words with this topic z 
                
                
                for reln in relns:
                    self.n_t_reln[y, self.reln2id[reln]] += 1
                self.n_d_t[d, y] += 1
                self.n_t_k[y, z] += 1 #count of topic k for theta role t
                self.n_t[y] += 1 #is it total count of "word" within theta role y. 
                
                #ZH added 
                self.n_t_w[y, word_id]+=1 #count of words for theta role y 

                word = Word(content=self.id2word[word_id], idx=word_id, z=z, y=y, relns=relns)
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
                    z, y, idx, relns = word.z, word.y, word.idx, word.relns
                
                    # subtract count for current topic and theta role; 
                    self.n_d_k[d, z] -= 1
                    self.n_k_w[z, idx] -= 1
                    self.n_t_w[y, idx] -=1
                    self.n_k[z] -= 1
                    for reln in relns:
                        self.n_t_reln[y, self.reln2id[reln]] -= 1
                    self.n_d_t[d, y] -= 1
                    self.n_t_k[y, z] -= 1
                    self.n_t[y] -= 1

                    # calculate conditional probability distribution for topic selection
                    # p(k|w,d) =~ p(w | k)* p(k | d) 
                    p_k = np.zeros(self.K)
                    for k in range(self.K):
                        p_w_k = (self.n_k_w[k, idx] + self.eta) / (self.n_k[k] + self.eta * self.V)#left side of the formula
                        p_k_d = (self.n_d_k[d, k] + self.alpha) / (N + self.K * self.alpha) #right side of the formula in the LDA..  N:Total number of tokens (words) in document i (i.e., the lentgh of the document:)
                        p_k[k] = p_k_d * p_w_k  

                    # select topic
                    z = np.random.choice(self.K, p=p_k / sum(p_k))
                    word.z = z
                    
                    # correct up to this point!!
                    # update counts for the selected topic
                    self.n_d_k[d, z] += 1
                    self.n_k_w[z, idx] += 1
                    self.n_t_w[y, idx] +=1
                    self.n_k[z] += 1

                    # probability of current that role,  given current grammatical relations, and for the selected topic z? 
                    # p(tr | reln, w, z, d) ~ p(w | tr) * p(reln | tr) * p(z | tr) *p(tr|d)
                    p_t_reln = np.zeros((self.T, self.R))
                    self.p_t_reln_w_z_d =  np.zeros((self.T, self.R))

                    for reln in relns:
                        for t in range(self.T):
                            #p(t|d)
                            p_t_d = (self.n_d_t[d, t] + self.gamma) / (N + self.T * self.gamma) #N is total number of theta role in document di, which is equal to the the length of document as each word is assigned a theta role to.
                            #p(reln|t)
                            p_reln_t = (self.n_t_reln[t, self.reln2id[reln]] + self.lam) \
                                 / (np.sum(self.n_t_reln[t, :] + self.lam * self.R)) # is n_t[t] = sum of all relns that are assigned to this theta role t
                            #p(z|d)? why do we need it here? we should not perhaps.?????????????????????????????????????????????
                            p_z_d = (self.n_d_k[d, z] + self.alpha) / (N + self.K * self.alpha)
                            #p(z|t)
                            #p_z_t = (self.n_t_k[t, z] + self.gamma) \
                            #     / (self.n_t[t] + self.gamma * self.T) # we defined a prior for topic distribution over theta role, named in omega
                            #p(z|t)
                            #p_z_t = (self.n_t_k[t, z] + self.omega) \
#                                  / (np.sum(self.n_t_k[t, :]) + self.omega * self.T) 
                            p_z_t = (self.n_t_k[t, z] ) \
                                 / (np.sum(self.n_t_k[t, :])) # we defined a prior for topic distribution over theta role, named in omega 
                            #Note: totlal number of topics for theta role t = np.sum(self.n_t_k[t, :])
                        #need the formula for p_z_d_t. what does it present? p(z,t|d) = p(z|d)p(z|t) or is it p(z|d, t)= p(z|d)p(z|t), i think it is the latter
                            
                            p_w_t = (self.n_t_w[t, idx] + self.etaprime) / (np.sum(self.n_t_w[t, :]) + self.etaprime * self.V)#
                            #now we have every part of p(t|reln, w, z, d)
                            self.p_t_reln_w_z_d[t, self.reln2id[reln]] = p_w_t * p_z_t * p_reln_t * p_t_d #what is the dimention of this one? 
                            #THE REST OF IT IS BASED ON CHACE'S CODE AND NOT SURE HOW TO ADOPT IT FOR MY APPRAOCH...
                            p_z_d_t = p_z_d * p_z_t
                            #p(t|reln) = ?
                            p_t_reln[t, self.reln2id[reln]] = p_t_d * p_reln_t * p_z_d_t
                            #[t,z] vs. [z, t]
                    
                    # maybe we should also care about and select theta role for each relation using probability distribution?
                    
                    # 
                    p_t = np.sum(p_t_reln, axis=1)
                    y = np.random.choice(self.T, p=p_t / np.sum(p_t))
                    word.y = y
                    
                    #ZHILA'S APPROACH: Only CHANGE chaces' code TO BE BASED ON "p_t_reln_w_z_d" that I created now? 
                    p_t = np.sum(self.p_t_reln_w_z_d, axis=1)
                    y = np.random.choice(self.T, p=p_t / np.sum(p_t))
                    word.y = y
                    
                    #z.y : follow the same strategy above to assign theta role for each topic... afterwards.. here is not working because we want to calcualte the matrix .... 
    
                    # update counts for selected theta role
                    for reln in relns:
                        self.n_t_reln[y, self.reln2id[reln]] += 1
                    self.n_d_t[d, y] += 1
                    self.n_t_k[y, z] += 1
                    self.n_t[y] += 1 #number of words assigned to this theta role y 
                    
                    #prob of all the theta role for each topic..
                    #prob of theta role for a word
                    
                    # p(t|w) = ???
                    #p(t|z) = (done below)
    
    
    
    
    #z.y: follow the same strategy above to assign theta role for each topic... afterwards..
    
    #after gips sampling              
    # p(reln|w) = p(reln|y)p(y|z)p(z|w) 

    def compute_theta(self):
        '''
        Compute the theta matrix from the plate diagram
        '''
        self.theta = np.zeros((self.D, self.K))
        for d in range(self.D):
            _N = len(self.corpus[d])
            self.theta[d] = (self.n_d_k[d] + self.alpha) / (_N + self.K * self.alpha)
        return self.theta

    def compute_phi(self):
        '''
        Compute the phi matrix from the plate diagram
        '''
        self.phi = np.zeros((self.D, self.T))
        for d in range(self.D):
            _N = len(self.corpus[d])
            self.phi[d] = (self.n_d_t[d] + self.gamma) / (_N + self.T * self.gamma) 
        return self.phi
    
    
        

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
        self.betaprime = np.zeros((self.T, self.V))
        for k in range(self.K):
            self.betaprime[k] = (self.n_t_w[k] + self.etaprime) / (self.n_k[k] + self.V * self.etaprime)
        return self.betaprime

    #Verify this one!!!
    def compute_zeta(self):
        '''
        Compute the zeta matrix from the plate diagram
        '''
        self.zeta = np.zeros((self.T, self.R))
        #?????for what reln does it get updated? shouldn't it be based on all the reln? shouldn't it be a 2 dimentioanl matrxi???
        for t in range(self.T):
            self.zeta[t] = (self.n_t_reln[t] + self.lam) / (self.n_t[t] + self.R * self.lam) 
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
# add the top documents as well

    def print_theta_roles(self):
        '''
        Print theta roles
        '''
        for t in range(self.T):
            reln_ids = np.argsort(self.zeta[t])[::-1][:10]
            probs = np.sort(self.zeta[t])[::-1][:10]
            top_relns = [self.vocab_relns[i] for i in reln_ids]
            strings = [f'{prob} * {reln}' for prob, reln in zip(probs, top_relns)]
            print(f"Theta Role {str(t)}: {', '.join(strings)}\n")

    def print_top_of_docs(self):
        '''
        Print top topics and theta roles for each document
        '''
        for d in range(self.D):
            top_topics = np.argsort(self.theta[d])[::-1][:3]
            top_theta = np.argsort(self.phi[d])[::-1][:3]
            print(f"Document {str(d)}: {' '.join(str(top_topics))} -- {' '.join(str(top_theta))}")
            
    def print_top_documents_topic(self, top_doc=20):
        '''
        print top documents for each topic
        '''
        
        for k in range(self.K):
            top_documents_indx = np.argsort(self.theta[:, k])[::-1][:top_doc]
            print('topic {0} \n'.format(k))
            try:
                for idx in top_documents_indx:
                    print("doc {0} - {1} - {2} \n ".format(idx, self.theta[idx, k], self.originaltext[str(idx)]))
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
            except:
                return top_documents_indx
    
    
    ###### NEXT, I need to calculate given a document what is the probability of each topic and each theta role for that document.
    def print_top_topics_doc(self, top_topics=20):
        print('This function is not yet done!!')
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
        print('This function is not yet done!!')
        for idx in range(self.D):
            top_thetas_indx = np.argsort(self.phi[idx, :][::-1][:top_thetas])
            print('document {0} \n'.format(idx))
            try:
                for t in top_thetas_indx:
                    print("theta {0} - {1} \n ".format(t, self.phi[idx, t]))
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
            except:
                return top_thetas_indx
    
    
    
    def print_top_documents_theta(self, top_doc=20):
        for t in range(self.T):
            top_documents_indx = np.argsort(self.phi[:, t])[::-1][:top_doc]
            print('topic number {0} \n'.format(t))
#             print('index of top documents: ', top_documents_indx)
            try:
                for idx in top_documents_indx:
                    print("doc {0} - {1} - {2} \n ".format(idx, self.phi[idx, t], self.originaltext[str(idx)]))
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
                print('------------------------------------------------------------------------')
            except:
                return top_documents_indx
                    
    
    def compute_matrices(self):
        _theta = self.compute_theta()
        _phi = self.compute_phi()
        _beta = self.compute_beta()
        _zeta = self.compute_zeta()

        return _theta, _phi, _beta, _zeta

    def print_all(self):
        self.print_topics()
        self.print_theta_roles()
#         self.print_top_of_docs()
    
    
    #     def compute_psi(self):
#         '''
#         Compute the psi matrix from the plate diagram
#         '''
#         self.psi = np.zeros((self.T, self.K))
#         #shouldn't it be a 2 dimentioanl matrxi??? no. it is going to be used to print top topics of each theta role
#         for t in range(self.T):
#             self.psi[t] = (self.n_t_k[t] + self.omega) / (self.n_t[t] + self.K * self.omega) 
#         return self.psi



