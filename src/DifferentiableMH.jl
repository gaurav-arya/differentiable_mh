module DifferentiableMH

using ArgCheck
using Distributions
using ForwardDiff
using LinearAlgebra
using PDMats
using StochasticAD
import Random
import Functors

include("dmh.jl")

export mh, mh_score
export mh_basic_kernel_init, mh_basic_kernel
export mh_kernel_init, mh_kernel, mh_f

include("abstract_proposal.jl")
export rand_proposal, logpdf_proposal, logratio_proposal, coupled_proposal
export MHProposalDistribution
export AbstractMHProposal, AbstractMHProposalCoupling

include("proposals/random_walk_mh_proposal.jl")
export RandomWalkMHProposal
export MaximumReflectionProposalCoupling

end