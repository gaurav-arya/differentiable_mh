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

include("proposals/independent_mh_proposal.jl")
include("proposals/random_walk_mh_proposal.jl")
export RandomWalkMHProposal, IndependentMHProposal
export MaximumReflectionProposalCoupling, IndependentMHProposalCoupling

# Experimental
include("proposals/papp_sherlock_coupling.jl")
include("proposals/mala_proposal.jl")
export MALAProposal, PappSherlockProposalCoupling

end
