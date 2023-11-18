import logging
import math
import json
import numpy as np
import torch
import torch.nn.functional as F 
import numpy as np 
import math 

from torch import nn


logger = logging.getLogger("detm")


class DETM(torch.nn.Module):
    def __init__(
            self,
            args,
            id2token,
            min_time,
            num_topics,
            windows,
            t_hidden_size,
            eta_hidden_size,
            eta_drop,
            rho_size,
            emb_size,
            enc_drop,
            eta_nlayers,
            delta,
            theta_act="relu",
            embeddings=None,
            device="cpu",
            adapt_embeddings=False
    ):
        super(DETM, self).__init__()
        self.min_time = min_time
        self.window_size = args.window_size
        self.device = device
        ## define hyperparameters
        self.num_topics = args.num_topics
        self.num_times = args.num_times
        self.vocab_size = args.vocab_size
        self.t_hidden_size = args.t_hidden_size
        self.eta_hidden_size = args.eta_hidden_size
        self.rho_size = args.rho_size
        self.emsize = args.emb_size
        self.enc_drop = args.enc_drop
        self.eta_nlayers = args.eta_nlayers
        self.t_drop = torch.nn.Dropout(args.enc_drop)
        self.delta = args.delta
        self.train_embeddings = args.train_embeddings

        self.theta_act = self.get_activation(args.theta_act)

        ## define the word embedding matrix \rho
        if args.train_embeddings:
            self.rho = torch.nn.Linear(args.rho_size, args.vocab_size, bias=False)
        else:
            num_embeddings, emsize = embeddings.size()
            rho = torch.nn.Embedding(num_embeddings, emsize)
            rho.weight.data = embeddings
            self.rho = rho.weight.data.clone().float().to(self.device)



        # #self.windows = windows
        # self.num_windows = len(windows)
        # self.num_times = len(windows) #args.num_times
        # #self.min_year = min_year
        # #self.years_per_window = years_per_window
        self.id2token = id2token
        # self.device = device
        # ## define hyperparameters
        # self.num_topics = num_topics

        # self.vocab_size = len(self.id2token) #args.vocab_size
        # self.t_hidden_size = t_hidden_size
        # self.eta_hidden_size = eta_hidden_size
        # self.rho_size = rho_size
        # self.emsize = emb_size
        # self.enc_drop = enc_drop
        # self.eta_nlayers = eta_nlayers
        # self.t_drop = torch.nn.Dropout(enc_drop)
        # self.delta = delta
        # self.train_embeddings = adapt_embeddings #args.train_embeddings

        # self.theta_act = self.get_activation(theta_act)

        ## define the word embedding matrix \rho
        #rho = torch.nn.Embedding(self.vocab_size, 300)
        #rho.weight.data = torch.tensor(embeddings)
        #self.rho = rho.weight.data.clone().float().to(self.device)

        #num_embeddings, emsize = embeddings.size()
        #rho = torch.nn.Embedding(num_embeddings, emsize)
        #rho.weight.data = embeddings
        #self.rho = rho.weight.data.clone().float().to(self.device)
    
        #self.rho = torch.nn.Linear(self.rho_size, self.vocab_size, bias=False)
        #embs = embeddings #torch.tensor(embeddings)
        #self.rho.weight.data = embs

        #if adapt_embeddings:
        #    self.rho.requires_grad_(True)
        #else:
        #    self.rho.requires_grad_(False)
            #print(self.rho, embs.shape)
            #with torch.no_grad():
            #    self.rho.weight.copy_(embs)
        #else:
        #    num_embeddings, emsize = embeddings.shape
        #    rho = torch.nn.Embedding(num_embeddings, emsize)
        #    rho.weight.data = torch.tensor(embeddings)
        #    self.rho = rho.weight.data.clone().float().to(device)
        #self.rho = torch.nn.Embedding.from_pretrained(torch.tensor(embeddings))
        #print(self.rho)
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        # self.mu_q_alpha = torch.nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size)).to(self.device)
        

        # self.logsigma_q_alpha = torch.nn.Parameter(torch.randn(self.num_topics, self.num_times, self.rho_size)).to(self.device)

        # ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        # self.q_theta = torch.nn.Sequential(
        #             torch.nn.Linear(self.vocab_size+self.num_topics, self.t_hidden_size), 
        #             self.theta_act,
        #             torch.nn.Linear(self.t_hidden_size, self.t_hidden_size),
        #             self.theta_act,
        #         )
        # self.mu_q_theta = torch.nn.Linear(self.t_hidden_size, self.num_topics, bias=True)
        # self.logsigma_q_theta = torch.nn.Linear(self.t_hidden_size, self.num_topics, bias=True)

        # ## define variational distribution for \eta via amortizartion... eta is K x T
        # self.q_eta_map = torch.nn.Linear(self.vocab_size, self.eta_hidden_size)
        # self.q_eta = torch.nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, eta_nlayers, dropout=eta_drop)
        # self.mu_q_eta = torch.nn.Linear(self.eta_hidden_size+self.num_topics, self.num_topics, bias=True)
        # self.logsigma_q_eta = torch.nn.Linear(self.eta_hidden_size+self.num_topics, self.num_topics, bias=True)
        ## define the variational parameters for the topic embeddings over time (alpha) ... alpha is K x T x L
        self.mu_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
        self.logsigma_q_alpha = nn.Parameter(torch.randn(args.num_topics, args.num_times, args.rho_size))
    
        ## define variational distribution for \theta_{1:D} via amortizartion... theta is K x D
        self.q_theta = nn.Sequential(
                    nn.Linear(args.vocab_size+args.num_topics, args.t_hidden_size), 
                    self.theta_act,
                    nn.Linear(args.t_hidden_size, args.t_hidden_size),
                    self.theta_act,
                )
        self.mu_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)
        self.logsigma_q_theta = nn.Linear(args.t_hidden_size, args.num_topics, bias=True)

        ## define variational distribution for \eta via amortizartion... eta is K x T
        self.q_eta_map = nn.Linear(args.vocab_size, args.eta_hidden_size)
        self.q_eta = nn.LSTM(args.eta_hidden_size, args.eta_hidden_size, args.eta_nlayers, dropout=args.eta_dropout)
        self.mu_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)
        self.logsigma_q_eta = nn.Linear(args.eta_hidden_size+args.num_topics, args.num_topics, bias=True)



        
    # def get_activation(self, act):
    #     if act == 'tanh':
    #         act = torch.nn.Tanh()
    #     elif act == 'relu':
    #         act = torch.nn.ReLU()
    #     elif act == 'softplus':
    #         act = torch.nn.Softplus()
    #     elif act == 'rrelu':
    #         act = torch.nn.RReLU()
    #     elif act == 'leakyrelu':
    #         act = torch.nn.LeakyReLU()
    #     elif act == 'elu':
    #         act = torch.nn.ELU()
    #     elif act == 'selu':
    #         act = torch.nn.SELU()
    #     elif act == 'glu':
    #         act = torch.nn.GLU()
    #     else:
    #         print('Defaulting to tanh activations...')
    #         act = torch.nn.Tanh()
    #     return act 

    # def reparameterize(self, mu, logvar):
    #     """Returns a sample from a Gaussian distribution via reparameterization.
    #     """
    #     if self.training:
    #         std = torch.exp(0.5 * logvar) 
    #         eps = torch.randn_like(std)
    #         return eps.mul_(std).add_(mu)
    #     else:
    #         return mu

    # def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
    #     """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
    #     """
    #     if p_mu is not None and p_logsigma is not None:

    #         # if torch.any(torch.isnan(q_logsigma)):
    #         #     print("q_logsigma has nans")
    #         # if torch.any(torch.isnan(q_mu)):
    #         #     print("q_mu has nans")                
    #         # if torch.any(torch.isnan(p_mu)):
    #         #     print("p_mu has nans")
    #         # if torch.any(torch.isnan(p_logsigma)):
    #         #     print("p_logsigma has nans")
                
    #         #q_logsigma = torch.nan_to_num(q_logsigma)
    #         #if torch.any(torch.isnan(q_logsigma)):
    #         #    print("q_logsigma still has nans")

    #         #p_logsigma = torch.nan_to_num(p_logsigma)
    #         #if torch.any(torch.isnan(p_logsigma)):
    #         #    print("p_logsigma still has nans")

    #         #q_mu = torch.nan_to_num(q_mu)
    #         #if torch.any(torch.isnan(q_mu)):
    #         #    print("q_mu still has nans")
                
    #         sigma_q_sq = torch.exp(q_logsigma)            
    #         #if torch.any(torch.isnan(sigma_q_sq)):
    #         #    print("sigma_q_sq has nans")
                
    #         sigma_p_sq = torch.exp(p_logsigma)
    #         #if torch.any(torch.isnan(sigma_p_sq)):
    #         #    print("sigma_p_sq has nans")

    #         #sigma_q_sq = torch.nan_to_num(sigma_q_sq)
    #         #if torch.any(torch.isnan(sigma_q_sq)):
    #         #    print("sigma_q_sq still has nans")                                
            
    #         #sigma_p_sq = torch.nan_to_num(sigma_p_sq)
    #         #if torch.any(torch.isnan(sigma_p_sq)):
    #         #    print("sigma_p_sq still has nans")

    #         #if torch.any(torch.isnan(p_mu)):
    #         #    print("p_mu still has nans")
            
    #         kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
    #         kl = kl - 1 + p_logsigma - q_logsigma
    #         kl = 0.5 * torch.sum(kl, dim=-1)
    #     else:
    #         kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
    #     return kl

    # def get_alpha(self): ## mean field
    #     alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
    #     kl_alpha = []
    #     alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])
    #     p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
    #     logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
    #     kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
    #     kl_alpha.append(kl_0)
    #     for t in range(1, self.num_times):
    #         alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            
    #         p_mu_t = alphas[t-1]
    #         logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))
    #         kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
    #         kl_alpha.append(kl_t)
    #     kl_alpha = torch.stack(kl_alpha).sum()
    #     return alphas, kl_alpha.sum()

    # def get_eta(self, rnn_inp): ## structured amortized inference
    #     inp = self.q_eta_map(rnn_inp).unsqueeze(1)
    #     hidden = self.init_hidden()
    #     output, _ = self.q_eta(inp, hidden)
    #     output = output.squeeze()

    #     etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
    #     kl_eta = []

    #     inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
    #     mu_0 = self.mu_q_eta(inp_0)
    #     logsigma_0 = self.logsigma_q_eta(inp_0)
    #     etas[0] = self.reparameterize(mu_0, logsigma_0)

    #     p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
    #     logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
    #     kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
    #     kl_eta.append(kl_0)
    #     for t in range(1, self.num_times):
    #         inp_t = torch.cat([output[t], etas[t-1]], dim=0)
    #         mu_t = self.mu_q_eta(inp_t)
    #         logsigma_t = self.logsigma_q_eta(inp_t)
    #         etas[t] = torch.nan_to_num(self.reparameterize(mu_t, logsigma_t))

    #         p_mu_t = etas[t-1]
    #         logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
    #         kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
    #         kl_eta.append(kl_t)
    #     kl_eta = torch.stack(kl_eta).sum()
    #     return etas, kl_eta

    # def get_theta(self, eta, bows, times): ## amortized inference
    #     """Returns the topic proportions.
    #     """
    #     eta_td = eta[times.type('torch.LongTensor')]
    #     inp = torch.cat([bows, eta_td], dim=1)
    #     q_theta = self.q_theta(inp)
    #     if self.enc_drop > 0:
    #         q_theta = self.t_drop(q_theta)
    #     mu_theta = self.mu_q_theta(q_theta)
    #     logsigma_theta = self.logsigma_q_theta(q_theta)
    #     z = self.reparameterize(mu_theta, logsigma_theta)
    #     theta = torch.nn.functional.softmax(z, dim=-1)
    #     kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))
    #     return theta, kl_theta

    # def get_beta(self, alpha):
    #     """Returns the topic matrix \beta of shape K x V
    #     """

    #     #logit = self.rho(alpha.view(alpha.size(0)*alpha.size(1), self.rho_size))

    #     tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
    #     logit = torch.mm(tmp, self.rho.permute(1, 0))
        
    #     logit = logit.view(alpha.size(0), alpha.size(1), -1)
    #     beta = torch.nn.functional.softmax(logit, dim=-1)
    #     return beta 

    # def get_nll(self, theta, beta, bows):
    #     theta = theta.unsqueeze(1)
    #     #theta = torch.nan_to_num(theta)
    #     #if torch.any(torch.isnan(theta)):
    #     #    print("theta has nans")
    #     #if torch.any(torch.isnan(beta)):
    #     #    print("beta has nans")

    #     loglik = torch.bmm(theta, beta).squeeze(1)
    #     #if torch.any(torch.isnan(loglik)):
    #     #    print("loglik has nans")
    #     #    #raise Exception("loglik: {}".format(beta))
    #     #loglik = loglik
    #     loglik = torch.log(loglik+1e-6)
    #     nll = -loglik * bows
    #     nll = nll.sum(-1)
    #     return nll  

    # def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
    #     #print(rnn_inp.shape, bows.shape, normalized_bows.shape, num_docs, times.shape)
    #     #print(bows.shape, bows.sum(1))
    #     #print(normalized_bows.shape, normalized_bows.sum(1))
    #     bsz = normalized_bows.size(0)
    #     # for doc, ndoc in zip(bows, normalized_bows):
    #     #     words = {}
    #     #     for i, c in enumerate(doc):
    #     #         if c > 0:
    #     #             print(self.id2token[i], c)
    #     #     break
    #     #     print(doc.sum())
    #     #print(bsz, num_docs)
    #     #sys.exit()

    #     coeff = num_docs / bsz
    #     alpha, kl_alpha = self.get_alpha()
    #     eta, kl_eta = self.get_eta(rnn_inp)
    #     theta, kl_theta = self.get_theta(eta, normalized_bows, times)
    #     kl_theta = kl_theta.sum() * coeff
    #     beta = self.get_beta(alpha)
    #     beta = beta[times.type('torch.LongTensor')]
    #     nll = self.get_nll(theta, beta, bows)
    #     nll = nll.sum() * coeff
    #     nelbo = nll + kl_alpha + kl_eta + kl_theta
    #     return nelbo, nll, kl_alpha, kl_eta, kl_theta

    # def init_hidden(self):
    #     """Initializes the first hidden state of the RNN used as inference network for \eta.
    #     """
    #     weight = next(self.parameters())
    #     nlayers = self.eta_nlayers
    #     nhid = self.eta_hidden_size
    #     return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
    def get_activation(self, act):
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'softplus':
            act = nn.Softplus()
        elif act == 'rrelu':
            act = nn.RReLU()
        elif act == 'leakyrelu':
            act = nn.LeakyReLU()
        elif act == 'elu':
            act = nn.ELU()
        elif act == 'selu':
            act = nn.SELU()
        elif act == 'glu':
            act = nn.GLU()
        else:
            print('Defaulting to tanh activations...')
            act = nn.Tanh()
        return act 

    def reparameterize(self, mu, logvar):
        """Returns a sample from a Gaussian distribution via reparameterization.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) 
            eps = torch.randn_like(std)
            return eps.mul_(std).add_(mu)
        else:
            return mu

    def get_kl(self, q_mu, q_logsigma, p_mu=None, p_logsigma=None):
        """Returns KL( N(q_mu, q_logsigma) || N(p_mu, p_logsigma) ).
        """
        if p_mu is not None and p_logsigma is not None:
            sigma_q_sq = torch.exp(q_logsigma)
            sigma_p_sq = torch.exp(p_logsigma)
            kl = ( sigma_q_sq + (q_mu - p_mu)**2 ) / ( sigma_p_sq + 1e-6 )
            kl = kl - 1 + p_logsigma - q_logsigma
            kl = 0.5 * torch.sum(kl, dim=-1)
        else:
            kl = -0.5 * torch.sum(1 + q_logsigma - q_mu.pow(2) - q_logsigma.exp(), dim=-1)
        return kl

    def get_alpha(self): ## mean field
        alphas = torch.zeros(self.num_times, self.num_topics, self.rho_size).to(self.device)
        kl_alpha = []

        alphas[0] = self.reparameterize(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :])

        p_mu_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics, self.rho_size).to(self.device)
        kl_0 = self.get_kl(self.mu_q_alpha[:, 0, :], self.logsigma_q_alpha[:, 0, :], p_mu_0, logsigma_p_0)
        kl_alpha.append(kl_0)
        for t in range(1, self.num_times):
            alphas[t] = self.reparameterize(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :]) 
            
            p_mu_t = alphas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics, self.rho_size).to(self.device))
            kl_t = self.get_kl(self.mu_q_alpha[:, t, :], self.logsigma_q_alpha[:, t, :], p_mu_t, logsigma_p_t)
            kl_alpha.append(kl_t)
        kl_alpha = torch.stack(kl_alpha).sum()
        return alphas, kl_alpha.sum()

    def get_eta(self, rnn_inp): ## structured amortized inference
        inp = self.q_eta_map(rnn_inp).unsqueeze(1)
        hidden = self.init_hidden()
        output, _ = self.q_eta(inp, hidden)
        output = output.squeeze()

        etas = torch.zeros(self.num_times, self.num_topics).to(self.device)
        kl_eta = []

        inp_0 = torch.cat([output[0], torch.zeros(self.num_topics,).to(self.device)], dim=0)
        mu_0 = self.mu_q_eta(inp_0)
        logsigma_0 = self.logsigma_q_eta(inp_0)
        etas[0] = self.reparameterize(mu_0, logsigma_0)

        p_mu_0 = torch.zeros(self.num_topics,).to(self.device)
        logsigma_p_0 = torch.zeros(self.num_topics,).to(self.device)
        kl_0 = self.get_kl(mu_0, logsigma_0, p_mu_0, logsigma_p_0)
        kl_eta.append(kl_0)
        for t in range(1, self.num_times):
            inp_t = torch.cat([output[t], etas[t-1]], dim=0)
            mu_t = self.mu_q_eta(inp_t)
            logsigma_t = self.logsigma_q_eta(inp_t)
            etas[t] = self.reparameterize(mu_t, logsigma_t)

            p_mu_t = etas[t-1]
            logsigma_p_t = torch.log(self.delta * torch.ones(self.num_topics,).to(self.device))
            kl_t = self.get_kl(mu_t, logsigma_t, p_mu_t, logsigma_p_t)
            kl_eta.append(kl_t)
        kl_eta = torch.stack(kl_eta).sum()
        return etas, kl_eta

    def get_theta(self, eta, bows, times): ## amortized inference
        """Returns the topic proportions.
        """
        eta_td = eta[times.type('torch.LongTensor')]
        inp = torch.cat([bows, eta_td], dim=1)
        q_theta = self.q_theta(inp)
        if self.enc_drop > 0:
            q_theta = self.t_drop(q_theta)
        mu_theta = self.mu_q_theta(q_theta)
        logsigma_theta = self.logsigma_q_theta(q_theta)
        z = self.reparameterize(mu_theta, logsigma_theta)
        theta = F.softmax(z, dim=-1)
        kl_theta = self.get_kl(mu_theta, logsigma_theta, eta_td, torch.zeros(self.num_topics).to(self.device))
        return theta, kl_theta

    def get_beta(self, alpha):
        """Returns the topic matrix \beta of shape K x V
        """
        if self.train_embeddings:
            logit = self.rho(alpha.view(alpha.size(0)*alpha.size(1), self.rho_size))
        else:
            tmp = alpha.view(alpha.size(0)*alpha.size(1), self.rho_size)
            logit = torch.mm(tmp, self.rho.permute(1, 0)) 
        logit = logit.view(alpha.size(0), alpha.size(1), -1)
        beta = F.softmax(logit, dim=-1)
        return beta 

    def get_nll(self, theta, beta, bows):
        theta = theta.unsqueeze(1)
        loglik = torch.bmm(theta, beta).squeeze(1)
        loglik = loglik
        loglik = torch.log(loglik+1e-6)
        nll = -loglik * bows
        nll = nll.sum(-1)
        return nll  

    def forward(self, bows, normalized_bows, times, rnn_inp, num_docs):
        bsz = normalized_bows.size(0)
        coeff = num_docs / bsz 
        alpha, kl_alpha = self.get_alpha()
        eta, kl_eta = self.get_eta(rnn_inp)
        theta, kl_theta = self.get_theta(eta, normalized_bows, times)
        kl_theta = kl_theta.sum() * coeff

        beta = self.get_beta(alpha)
        beta = beta[times.type('torch.LongTensor')]
        nll = self.get_nll(theta, beta, bows)
        nll = nll.sum() * coeff
        nelbo = nll + kl_alpha + kl_eta + kl_theta
        return nelbo, nll, kl_alpha, kl_eta, kl_theta

    def init_hidden(self):
        """Initializes the first hidden state of the RNN used as inference network for \eta.
        """
        weight = next(self.parameters())
        nlayers = self.eta_nlayers
        nhid = self.eta_hidden_size
        return (weight.new_zeros(nlayers, 1, nhid), weight.new_zeros(nlayers, 1, nhid))
